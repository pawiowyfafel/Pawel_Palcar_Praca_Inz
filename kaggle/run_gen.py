"""Generacja na Kaggle GPU (T4/P100): obrazy AI (t2i, i2i) i inpainting dyfuzyjny wg jobs.csv.

Uruchamiany w notebooku Kaggle z katalogu datasetu wejściowego:
    python run_gen.py --gen sdxl [--hours 11] [--dry-run] [--input KATALOG_Z_jobs.csv]
Zamienniki generatora testowego (decyzja 2026-10-14: PixArt-Σ zamiast FLUX, bo FLUX wymaga logowania
na HF): --gen pixart / kandinsky wykonują zadania "flux"; PixArt nie ma img2img w diffusers,
więc zadania img2img robi jako text2img z tym samym promptem (bez obrazu źródłowego).
Wynik: /kaggle/working/<gen>/img/*.jpg + meta.jsonl, spakowane do /kaggle/working/<gen>.zip.
Wznawia się po przerwaniu (pomija id obecne w meta.jsonl).
"""
import argparse, csv, json, shutil, time
from pathlib import Path

IN = Path(__file__).resolve().parent
NEG = ("painting, illustration, cartoon, anime, drawing, render, cgi, 3d, text, watermark, logo, "
       "frame, border, deformed")
MODELS = {
    "sd15":    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "sdxl":    "stabilityai/stable-diffusion-xl-base-1.0",
    "flux":    "black-forest-labs/FLUX.1-schnell",
    "kandinsky": "kandinsky-community/kandinsky-2-2-decoder",   # zapas, gdyby FLUX nie działał na T4
    "pixart":  "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",          # zamiast FLUX (bez logowania na HF)
    "sd15inp": "stable-diffusion-v1-5/stable-diffusion-inpainting",
    "sdxlinp": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
}
STEPS = {"sd15": 25, "sdxl": 25, "flux": 4, "kandinsky": 30, "pixart": 20, "sd15inp": 30, "sdxlinp": 30}
LONG = {"sd15": 768, "sd15inp": 768, "sdxl": 1024, "sdxlinp": 1024, "flux": 1024, "kandinsky": 1024,
        "pixart": 1024}
STAND_IN = {"kandinsky": "flux", "pixart": "flux"}     # zamiennik -> czyje zadania wykonuje
NO_I2I = {"pixart"}


def dims(w, h, long_side):
    s = long_side / max(w, h)
    return int(w * s) // 8 * 8, int(h * s) // 8 * 8


def load(gen):
    import torch
    from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, AutoPipelineForInpainting
    fp16 = dict(torch_dtype=torch.float16)
    if gen in ("sd15inp", "sdxlinp"):
        p = AutoPipelineForInpainting.from_pretrained(MODELS[gen], **fp16, variant="fp16" if gen == "sdxlinp" else None)
        return {"inpaint": p.to("cuda")}
    if gen == "pixart":
        from diffusers import PixArtSigmaPipeline
        p = PixArtSigmaPipeline.from_pretrained(MODELS[gen], **fp16)
        p.enable_model_cpu_offload()                       # T5-XXL ~9.5 GB w fp16
        return {"t2i": p}
    if gen == "flux":
        from diffusers import FluxTransformer2DModel, BitsAndBytesConfig
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        tr = FluxTransformer2DModel.from_pretrained(MODELS[gen], subfolder="transformer", quantization_config=q,
                                                    torch_dtype=torch.float16)
        t2i = AutoPipelineForText2Image.from_pretrained(MODELS[gen], transformer=tr, torch_dtype=torch.float16)
        t2i.enable_model_cpu_offload()
        return {"t2i": t2i, "i2i": AutoPipelineForImage2Image.from_pipe(t2i)}
    kw = dict(fp16)
    if gen == "sdxl":
        from diffusers import AutoencoderKL
        kw["vae"] = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    t2i = AutoPipelineForText2Image.from_pretrained(MODELS[gen], **kw).to("cuda")
    return {"t2i": t2i, "i2i": AutoPipelineForImage2Image.from_pipe(t2i)}


def run(pipes, gen, j):
    import torch
    from PIL import Image
    g = torch.Generator("cpu").manual_seed(int(j["seed"]))
    common = dict(prompt=j["prompt"], num_inference_steps=STEPS[gen], generator=g)
    if gen != "flux":
        common["negative_prompt"] = NEG
    else:
        common["guidance_scale"] = 0.0
    if j["mode"] == "t2i":
        w, h = (int(j["w"]), int(j["h"]))
        return pipes["t2i"](width=w, height=h, **common).images[0]
    src = Image.open(IN / "src" / j["src"]).convert("RGB")
    w, h = dims(*src.size, LONG[gen])
    src = src.resize((w, h), Image.LANCZOS)
    if j["mode"] == "i2i":
        return pipes["i2i"](image=src, strength=float(j["strength"]), **common).images[0]
    mask = Image.open(IN / "masks" / j["mask"]).convert("L").resize((w, h), Image.NEAREST)
    out = pipes["inpaint"](image=src, mask_image=mask, width=w, height=h, strength=0.99, **common).images[0]
    return out.resize((w, h), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True, choices=list(MODELS))
    ap.add_argument("--hours", type=float, default=11.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--input", default=None, help="katalog z jobs.csv/src/masks (domyślnie obok skryptu)")
    ap.add_argument("--limit", type=int, default=None, help="tylko N zadań (sesja testowa)")
    ap.add_argument("--exclude", default=None, help="plik z id zadań zrobionych w poprzedniej sesji")
    a = ap.parse_args()
    global IN
    if a.input:
        IN = Path(a.input)
    jobs = [j for j in csv.DictReader(open(IN / "jobs.csv", encoding="utf-8"))
            if j["gen"] == STAND_IN.get(a.gen, a.gen)]
    if a.gen in NO_I2I:                                    # img2img -> text2img, własna grupa
        sizes = [(1024, 768), (768, 1024)]
        jobs = [j if j["mode"] == "t2i" else
                {**j, "mode": "t2i", "src": "", "strength": "", "base_id": "", "group": f"kaggle:{j['id']}",
                 "w": sizes[int(j["seed"]) % 2][0], "h": sizes[int(j["seed"]) % 2][1]} for j in jobs]
    out = Path("/kaggle/working") / a.gen if Path("/kaggle/working").exists() else IN / "out" / a.gen
    (out / "img").mkdir(parents=True, exist_ok=True)
    mf = out / "meta.jsonl"
    done = {json.loads(l)["id"] for l in open(mf, encoding="utf-8")} if mf.exists() else set()
    if a.exclude:                                          # zrobione w poprzedniej sesji
        done |= set(Path(a.exclude).read_text().split())
    todo = [j for j in jobs if j["id"] not in done][:a.limit]
    print(f"{a.gen}: zadań {len(jobs)}, zrobione {len(done)}, do zrobienia {len(todo)}")
    if a.dry_run:
        for j in todo[:5]:
            print(j)
        return
    import numpy as np
    pipes, t_end, skipped = load(a.gen), time.time() + a.hours * 3600, 0
    with open(mf, "a", encoding="utf-8") as f:
        for k, j in enumerate(todo):
            if time.time() > t_end:
                print("limit czasu – przerywam, wynik częściowy")
                break
            t = time.time()
            # czarny obraz: filtr NSFW w SD 1.5 (fałszywe alarmy na krajobrazach) albo NaN w fp16
            # -> 2 próby z innym ziarnem, potem pomiń; seria pustych obrazów = generator nie działa
            for attempt in range(3):
                im = run(pipes, a.gen, {**j, "seed": int(j["seed"]) + attempt})
                if np.asarray(im).std() >= 2:
                    break
            else:
                skipped += 1
                print(f"pominięto {j['id']} (pusty obraz po 3 próbach)", flush=True)
                if skipped >= 10 and skipped > 0.2 * (k + 1):
                    raise SystemExit(f"za dużo pustych obrazów – generator {a.gen} nie działa na tym GPU")
                continue
            j = {**j, "seed": int(j["seed"]) + attempt}
            im.save(out / "img" / f"{j['id']}.jpg", quality=95)
            f.write(json.dumps({**j, "gen_used": a.gen, "out_w": im.width, "out_h": im.height,
                                "time_s": round(time.time() - t, 1)}, ensure_ascii=False) + "\n")
            f.flush()
            if k % 10 == 0:
                print(f"{k + 1}/{len(todo)}  {time.time() - t:.1f} s/obraz", flush=True)
    shutil.make_archive(str(out), "zip", out)
    print("gotowe:", str(out) + ".zip")


if __name__ == "__main__":
    main()

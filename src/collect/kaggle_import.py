"""Import wyników z Kaggle: wrzuć pobrane <gen>.zip do data/kaggle/out/ i uruchom ten skrypt.

ai      -> Meta("kaggle_ai"),      subclass "ai"
inpaint -> Meta("kaggle_inpaint"), subclass "manipulated" (manip="diffusion_inpaint")
img2img/inpainting dziedziczą group po zdjęciu źródłowym (ten sam split); text2img ma własną grupę
i wymuszony split = rola zadania (force_split). test_ood -> holdout="generator".
"""
import json, zipfile
from common import Meta, DATA

IN = DATA / "kaggle" / "out"


def main():
    metas = {"ai": Meta("kaggle_ai"), "inpaint": Meta("kaggle_inpaint")}
    zips = sorted(IN.glob("*.zip"))
    print("paczki:", [z.name for z in zips])
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            for line in zf.read("meta.jsonl").decode("utf-8").splitlines():
                j = json.loads(line)
                meta = metas[j["task"]]
                if meta.has(j["id"]):
                    continue
                rec = dict(source_id=f"{j['gen_used']}:{j['id']}", group=j["group"], label=0,
                           gen=j["gen_used"], gen_mode=j["mode"], prompt=j["prompt"], role=j["role"],
                           seed=int(j["seed"]), base_id=j["base_id"] or None,
                           license=f"wygenerowane ({j['gen_used']}), licencja modelu",
                           holdout="generator" if j["role"] == "test_ood" else None,
                           force_split=j["role"] if j["mode"] == "t2i" else None)
                if j["task"] == "ai":
                    rec.update(subclass="ai", strength=float(j["strength"]) if j["strength"] else None)
                else:
                    rec.update(subclass="manipulated", manip="diffusion_inpaint")
                meta.add(j["id"], zf.read(f"img/{j['id']}.jpg"), **rec)
    for m in metas.values():
        m.close()


if __name__ == "__main__":
    IN.mkdir(parents=True, exist_ok=True)
    main()

"""Eksport do LiteRT (TFLite) przez ai-edge-torch – tylko Linux (Kaggle/Colab), w OSOBNYM środowisku wirtualnym,
bo ai-edge-torch wymaga konkretnej wersji torch.

    python -m train.litert_export <run> [<run> ...] [--calib-dir <katalog z .npy>]

Dla każdego runu do <run>/export/: model_fp32.tflite, model_fp16.tflite, model_int8.tflite (PT2E, statyczna,
kalibracja na zapisanych wejściach) – każdy krok osobno w try/except (API zmienia się między wersjami),
wynik i błędy w <run>/export/litert.json. Zgodność z PyTorch: średnia |różnica| logitów na wejściach kalibracyjnych.
"""
import argparse, json, traceback
from pathlib import Path
import numpy as np, torch
from train.data import CLASSES
from train.models import build

try:                                                  # od 2025 pakiet nazywa się litert-torch; ai-edge-torch to
    import litert_torch as aet                        # już tylko pusta "wydmuszka" bez convert()
except ImportError:
    import ai_edge_torch as aet


def load(run):
    cfg = json.loads((Path(run) / "config.json").read_text(encoding="utf-8"))
    model, _, _, _ = build(cfg["model"], 1 if cfg["head"] == "binary" else len(CLASSES), pretrained=False)
    model.load_state_dict(torch.load(Path(run) / "best.pt", map_location="cpu"))
    return cfg, model.eval()


def try_step(rep, key, fn):
    try:
        rep[key] = fn()
        print(key, "OK", rep[key], flush=True)
    except Exception as e:
        rep[key] = {"blad": f"{type(e).__name__}: {e}"[:500], "trace": traceback.format_exc()[-1500:]}
        print(key, "BŁĄD", rep[key]["blad"], flush=True)


def tflite_out(path, x):
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=str(path))
    it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    si, zi = i.get("quantization", (0.0, 0))
    so, zo = o.get("quantization", (0.0, 0))
    outs = []
    for b in x:
        v = b[None].numpy()
        if np.issubdtype(i["dtype"], np.integer) and si:          # wejście skwantyzowane: float -> int
            v = np.clip(np.round(v / si + zi), np.iinfo(i["dtype"]).min, np.iinfo(i["dtype"]).max)
        it.set_tensor(i["index"], v.astype(i["dtype"]))
        it.invoke()
        r = it.get_tensor(o["index"])[0].astype(np.float32)
        if np.issubdtype(o["dtype"], np.integer) and so:          # wyjście skwantyzowane: int -> float
            r = (r - zo) * so
        outs.append(r)
    return np.stack(outs)


def io_info(path):
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=str(path))
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    return {"wejscie": str(np.dtype(i["dtype"])), "wyjscie": str(np.dtype(o["dtype"]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--calib", default=None, help="plik .npy z wejściami kalibracyjnymi (N×3×H×W), z export.py")
    ap.add_argument("--data-root", default=None, help="ocena ROC-AUC na 1000 zdjęć testowych (te same co w export.py)")
    a = ap.parse_args()
    for run in a.runs:
        cfg, model = load(run)
        out = Path(run) / "export"
        out.mkdir(exist_ok=True)
        size = cfg["img_size"]
        calib_f = out / "calib.npy"
        x = torch.from_numpy(np.load(calib_f)) if calib_f.exists() else torch.randn(32, 3, size, size)
        sample = (x[:1],)
        with torch.no_grad():
            ref = model(x).numpy()
        rep = {"pakiet": aet.__name__, "kalibracja": "export/calib.npy" if calib_f.exists() else "losowe"}

        def fp32():
            p = out / "model_fp32.tflite"
            aet.convert(model, sample).export(str(p))
            return {"MB": round(p.stat().st_size / 2 ** 20, 2), "max_abs_diff": float(np.abs(tflite_out(p, x) - ref).max())}

        def via_quantizer(name, recipe_names):
            """FP16 / dynamiczny INT8 oficjalnym narzędziem LiteRT (ai-edge-quantizer) z gotowego pliku FP32."""
            from ai_edge_quantizer import quantizer, recipe
            src, p = out / "model_fp32.tflite", out / f"model_{name}.tflite"
            for rn in recipe_names:                                   # nazwy receptur różnią się między wersjami
                if hasattr(recipe, rn):
                    qt = quantizer.Quantizer(str(src))
                    qt.load_quantization_recipe(getattr(recipe, rn)())
                    qt.quantize().export_model(str(p), overwrite=True)
                    return {"receptura": rn, "MB": round(p.stat().st_size / 2 ** 20, 2), **io_info(p),
                            "max_abs_diff": float(np.abs(tflite_out(p, x) - ref).max())}
            raise RuntimeError(f"brak receptur {recipe_names} w ai_edge_quantizer.recipe: {dir(recipe)}")

        def fp16_old():
            import tensorflow as tf
            p = out / "model_fp16.tflite"
            # konwerter ustawia atrybuty przez setattr -> target_spec musi być obiektem TargetSpec, nie słownikiem
            flags = {"optimizations": [tf.lite.Optimize.DEFAULT],
                     "target_spec": tf.lite.TargetSpec(supported_types=[tf.float16])}
            try:                                      # nazwa parametru zależy od wersji pakietu
                conv = aet.convert(model, sample, _litert_converter_flags=flags)
            except TypeError:
                conv = aet.convert(model, sample, _ai_edge_converter_flags=flags)
            conv.export(str(p))
            return {"MB": round(p.stat().st_size / 2 ** 20, 2), "max_abs_diff": float(np.abs(tflite_out(p, x) - ref).max())}

        def int8():
            try:                                      # nowsze torch: PT2E przeniesione do pakietu torchao
                from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
            except ImportError:
                from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
            pkg = aet.__name__
            q = __import__(f"{pkg}.quantize.pt2e_quantizer", fromlist=["x"])
            qc = __import__(f"{pkg}.quantize.quant_config", fromlist=["x"])
            quantizer = q.PT2EQuantizer().set_global(q.get_symmetric_quantization_config(is_per_channel=True))
            exp = getattr(torch.export, "export_for_training", torch.export.export)
            m = exp(model, sample).module()
            m = prepare_pt2e(m, quantizer)
            with torch.no_grad():
                for b in x.split(1):                  # eksport ma ustaloną paczkę = 1
                    m(b)
            m = convert_pt2e(m, fold_quantize=False)
            p = out / "model_int8.tflite"
            aet.convert(m, sample, quant_config=qc.QuantConfig(pt2e_quantizer=quantizer)).export(str(p))
            return {"MB": round(p.stat().st_size / 2 ** 20, 2), "max_abs_diff": float(np.abs(tflite_out(p, x) - ref).max())}

        def int8dyn():
            """Dynamiczny INT8: wagi INT8, aktywacje float (kwantyzowane w locie) – bez kalibracji."""
            import tensorflow as tf
            p = out / "model_int8dyn.tflite"
            flags = {"optimizations": [tf.lite.Optimize.DEFAULT]}
            try:
                conv = aet.convert(model, sample, _litert_converter_flags=flags)
            except TypeError:
                conv = aet.convert(model, sample, _ai_edge_converter_flags=flags)
            conv.export(str(p))
            return {"MB": round(p.stat().st_size / 2 ** 20, 2), "max_abs_diff": float(np.abs(tflite_out(p, x) - ref).max())}

        def int8_checked():
            r = int8()
            r.update(io_info(out / "model_int8.tflite"))
            return r

        # FP16 pominięte: ai-edge-quantizer nie ma receptury FP16, a flagi konwertera litert-torch są ignorowane;
        # delegat GPU LiteRT i tak liczy model FP32 w FP16 (domyślnie dopuszczona utrata precyzji).
        steps = (("fp32", fp32),
                 ("int8dyn", lambda: via_quantizer("int8dyn", ["dynamic_wi8_afp32", "dynamic_legacy_wi8_afp32"])),
                 ("int8", int8_checked))
        for k, fn in steps:
            try_step(rep, k, fn)
        if a.data_root:                                   # ROC-AUC na tych samych 1000 zdjęciach co ONNX
            from sklearn.metrics import roc_auc_score
            from train.export import batches, p_violation
            _, _, mean, std = build(cfg["model"], 1, pretrained=False)
            cfg["data_root"] = a.data_root
            df, chk = batches(cfg, "test", 1000, size, mean, std, seed=1)
            y = (df.label == 0).astype(int).values
            with torch.no_grad():
                rep["pytorch_roc_auc"] = float(roc_auc_score(y, p_violation(model(chk), cfg["head"])))
            for k in ("fp32", "int8dyn", "int8"):
                f = out / f"model_{k}.tflite"
                if f.exists() and "blad" not in rep.get(k, {}):
                    try:
                        rep[k]["roc_auc"] = float(roc_auc_score(y, p_violation(tflite_out(f, chk), cfg["head"])))
                    except Exception as e:
                        rep[k]["roc_auc_blad"] = str(e)[:300]
            print("ROC-AUC:", {k: rep[k].get("roc_auc") for k in ("fp32", "int8dyn", "int8") if isinstance(rep.get(k), dict)})
        (out / "litert.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

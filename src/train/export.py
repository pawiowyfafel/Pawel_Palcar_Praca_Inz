"""Eksport wytrenowanego modelu na telefon + kwantyzacja + sprawdzenie zgodności z PyTorch.

    python -m train.export runs/effl0_multi_s0 [--calib 200] [--check 500] [--litert]

Ścieżka podstawowa (działa wszędzie, także na Windows): PyTorch -> ONNX (FP32) -> ONNX FP16 i ONNX INT8
(kwantyzacja statyczna, kalibracja na N obrazach z train) -> uruchamiane przez ONNX Runtime Mobile na Androidzie.
Ścieżka LiteRT/TFLite (--litert, tylko Linux, np. Kaggle/Colab): pakiet ai-edge-torch (nazwa i API zmieniają się
między wersjami – sprawdzić przy pierwszym eksporcie). Wynik: <run>/export/*.onnx|*.tflite + export.json
(rozmiar pliku, zgodność p_violation z PyTorch: średni |różnica|, ROC-AUC przed/po).
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
from train.data import CLASSES, TrailDataset, transforms
from train.models import build


def load(run, data_root=None):
    cfg = json.loads((Path(run) / "config.json").read_text(encoding="utf-8"))
    if data_root:                                         # inna sesja = inna ścieżka montowania datasetu
        cfg["data_root"] = data_root
    nc = 1 if cfg["head"] == "binary" else len(CLASSES)
    model, _, mean, std = build(cfg["model"], nc, pretrained=False)
    model.load_state_dict(torch.load(Path(run) / "best.pt", map_location="cpu"))
    return cfg, model.eval(), cfg["img_size"], mean, std


def batches(cfg, split, n, size, mean, std, seed=0):
    root = Path(cfg["data_root"])
    df = pd.read_parquet(root / cfg["manifest"])
    df = df[df.split == split].sample(min(n, (df.split == split).sum()), random_state=seed).reset_index(drop=True)
    ds = TrailDataset(df, root, transforms(size, mean, std, False), cfg["head"])
    return df, torch.stack([ds[i][0] for i in range(len(ds))])


def p_violation(logits, head):
    logits = torch.as_tensor(logits, dtype=torch.float32)
    return (torch.sigmoid(logits[:, 0]) if head == "binary" else 1 - logits.softmax(1)[:, 0]).numpy()


def export_onnx(model, size, out, calib_x):
    import onnx
    from onnxconverter_common import float16
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process
    f32 = out / "model_fp32.onnx"
    torch.onnx.export(model, torch.zeros(1, 3, size, size), f32, input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "n"}, "logits": {0: "n"}}, opset_version=17, dynamo=False)
    onnx.save(float16.convert_float_to_float16(onnx.load(f32), keep_io_types=True), out / "model_fp16.onnx")

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(calib_x.split(1))

        def get_next(self):
            x = next(self.it, None)
            return None if x is None else {"image": x.numpy()}

    pre = out / "model_fp32_pre.onnx"
    quant_pre_process(str(f32), str(pre))
    quantize_static(str(pre), str(out / "model_int8.onnx"), Reader(), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True)
    pre.unlink()
    return {"fp32": f32, "fp16": out / "model_fp16.onnx", "int8": out / "model_int8.onnx"}


def run_onnx(path, x):
    import onnxruntime as ort
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return np.concatenate([s.run(None, {"image": b.numpy()})[0] for b in x.split(32)])


def export_litert(model, size, out, calib_x):
    """LiteRT (TFLite) przez ai-edge-torch: FP32 i dynamiczna INT8 wag. Do weryfikacji na Linuksie."""
    import ai_edge_torch
    sample = (torch.zeros(1, 3, size, size),)
    ai_edge_torch.convert(model, sample).export(str(out / "model_fp32.tflite"))
    tfl = {"fp32": out / "model_fp32.tflite"}
    try:
        from ai_edge_torch.generative.quantize import quant_recipes
        q = ai_edge_torch.convert(model, sample, quant_config=quant_recipes.full_int8_dynamic_recipe())
        q.export(str(out / "model_int8dyn.tflite"))
        tfl["int8dyn"] = out / "model_int8dyn.tflite"
    except Exception as e:                                   # API kwantyzacji zmienia się między wersjami
        print("kwantyzacja LiteRT nieudana:", e)
    return tfl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--calib", type=int, default=200)
    ap.add_argument("--check", type=int, default=500)
    ap.add_argument("--litert", action="store_true")
    ap.add_argument("--data-root", default=None)
    a = ap.parse_args()
    cfg, model, size, mean, std = load(a.run, a.data_root)
    out = Path(a.run) / "export"
    out.mkdir(exist_ok=True)
    _, calib_x = batches(cfg, "train", a.calib, size, mean, std)
    np.save(out / "calib.npy", calib_x[:64].numpy())      # wejścia kalibracyjne dla litert_export.py
    chk_df, chk_x = batches(cfg, "test", a.check, size, mean, std, seed=1)
    y = (chk_df.label == 0).astype(int).values
    with torch.no_grad():
        ref = p_violation(torch.cat([model(b) for b in chk_x.split(32)]), cfg["head"])
    report = {"pytorch": {"roc_auc": float(roc_auc_score(y, ref)) if 0 < y.sum() < len(y) else None}}
    files = export_onnx(model, size, out, calib_x)
    for k, f in files.items():
        p = p_violation(run_onnx(f, chk_x), cfg["head"])
        report[f"onnx_{k}"] = {"MB": round(f.stat().st_size / 2 ** 20, 2), "mean_abs_diff": float(np.abs(p - ref).mean()),
                               "roc_auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None}
    if a.litert:
        for k, f in export_litert(model, size, out, calib_x).items():
            report[f"litert_{k}"] = {"MB": round(f.stat().st_size / 2 ** 20, 2)}
    (out / "export.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

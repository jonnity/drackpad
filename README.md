# drackpad

紙とカメラを使って机の上をトラックパッド化するツール（開発中）。
全体計画は [`spec.md`](./spec.md) を参照。

## 現在の状態

**Milestone 0（スパイク）完了。** `ort`（ONNX Runtime）で手のランドマーク
`.onnx` モデルを読み込み、1 枚の画像から 21 点のランドマーク（人差し指の
先端を含む）の座標を取り出して標準出力に表示するところまで動作する。

## セットアップと実行

ONNX Runtime の共有ライブラリとモデルファイルは大きく・プラットフォーム
依存のためリポジトリには含めていない。セットアップスクリプトで取得する：

```sh
./scripts/setup.sh
```

スクリプトが表示する手順に従って実行する：

```sh
export ORT_DYLIB_PATH="$(pwd)/vendor/onnxruntime-linux-x64-1.20.1/lib/libonnxruntime.so"
cargo run                       # testdata/ 内の画像すべてを処理
cargo run -- path/to/hand.jpg   # 任意の画像を指定
```

出力例（`testdata/pointing_up.jpg`、人差し指を上に立てた手）：

```
[testdata/pointing_up.jpg] 358x376  score=0.994 handedness=1.000
    8 (index_finger_tip): x=171.7 y=72.7 z=-12.857
```

## 技術メモ

- **推論バックエンド:** [`ort`](https://ort.pyke.io/) 2.0 rc。
  `load-dynamic` 機能を使い、実行時に `ORT_DYLIB_PATH` から ONNX Runtime を
  読み込む。`Cargo.toml` で固定している API レベル（`api-20`）は
  ONNX Runtime のマイナーバージョン（1.**20**.x）と一致させる必要がある。
- **モデル:** MediaPipe Hands のランドマークモデルを ONNX 化したもの
  （[PINTO0309/hand-gesture-recognition-using-onnx](https://github.com/PINTO0309/hand-gesture-recognition-using-onnx)）。
  入力 `[N, 3, 224, 224]` RGB（0.0–1.0）、出力は 21 点 × (x, y, z) を
  224×224 入力ピクセル座標で返す。これを元画像サイズへスケールし直している。
- **テスト画像:** `testdata/` は MediaPipe の公開アセット。

### コードの構成

- `src/landmark.rs` — `HandLandmarker`: 前処理（リサイズ・正規化）、推論、
  出力デコード（座標スケール変換）。
- `src/main.rs` — CLI。モデルと画像を読み込み、ランドマークを表示する。

# drackpad

紙とカメラを使って机の上をトラックパッド化するツール（開発中）。
全体計画は [`spec.md`](./spec.md) を参照。

## 現在の状態

**Milestone 0（スパイク）完了 + Milestone 1 の入力基盤に着手。**
`ort`（ONNX Runtime）で手のランドマーク `.onnx` モデルを読み込み、
21 点のランドマーク（人差し指の先端を含む）の座標を取り出して表示する。
フレームの入力は **OpenCV** 経由：静止画（`cv::imread`）と
カメラ（`cv::VideoCapture`）の 2 系統に対応している。

## 事前準備（システムライブラリ）

`opencv` クレートは OpenCV 本体と libclang を必要とする。

```sh
# Debian / Ubuntu
sudo apt-get install -y libopencv-dev clang libclang-dev

# macOS (Homebrew)
brew install opencv
```

## セットアップと実行

ONNX Runtime の共有ライブラリとモデルファイルは大きく・プラットフォーム
依存のためリポジトリには含めていない。セットアップスクリプトで取得する：

```sh
./scripts/setup.sh
```

スクリプトが表示する手順に従って実行する：

```sh
export ORT_DYLIB_PATH="$(pwd)/vendor/onnxruntime-linux-x64-1.20.1/lib/libonnxruntime.so"

# 静止画（OpenCV の cv::imread で読み込む）
cargo run                       # testdata/ 内の画像すべてを処理
cargo run -- path/to/hand.jpg   # 任意の画像を指定

# カメラ（cv::VideoCapture、実機のみ）
cargo run -- --camera           # デバイス 0 を 30 フレーム処理
cargo run -- --camera 1 100     # デバイス 1 を 100 フレーム処理
```

出力例（`testdata/pointing_up.jpg`、人差し指を上に立てた手）：

```
[testdata/pointing_up.jpg] 358x376  score=0.994 handedness=1.000
    8 (index_finger_tip): x=171.8 y=72.7 z=-13.461
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

- `src/capture.rs` — OpenCV によるフレーム取得。`imread_bgr`（静止画）、
  `Camera`（`VideoCapture` ラッパ）、`mat_to_input`（BGR `Mat` →
  `[1,3,224,224]` RGB テンソルへの前処理）。
- `src/landmark.rs` — `HandLandmarker`: 推論と出力デコード（224×224 座標を
  元画像サイズへスケール変換）。
- `src/main.rs` — CLI。静止画／カメラを OpenCV で読み込み、ランドマークを
  表示する。

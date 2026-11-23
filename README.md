# HocMay Image Inpainting Demo

Ứng dụng web để chạy và so sánh ba mô hình xóa vật thể/phục hồi vùng ảnh bị khuyết: **Dilated U-Net**, **DeepFillv1** và **LaMa FFC**. Backend dùng FastAPI + PyTorch; giao diện thuần HTML/CSS/JavaScript.

Demo nhận ảnh và mask, chạy các mô hình được chọn, ghép phần dự đoán vào vùng khuyết, rồi trả ảnh kết quả, heatmap sai số và các chỉ số L1/PSNR/SSIM. Ảnh được center-crop và resize về 256×256 trước khi suy luận.

> LaMa trong dự án là một triển khai FFC gọn cho bài tập này; đây không phải bản phát hành chính thức của dự án LaMa.

## Mô hình và checkpoint

| ID dùng trong API | Kiến trúc | Tệp trọng số |
| --- | --- | --- |
| `dilated_unet` | Dilated U-Net + PatchGAN | `dilated_unet_baseline/train_results/ckpt_best.pt` |
| `deepfillv1` | DeepFillv1 coarse-to-fine + contextual attention | `deepfillv1_ducminh05/train_results/ckpt_epoch16.pt` |
| `lama_ffc` | LaMa FFC gọn, 9 FFC residual blocks | `lama_ffc/train_results/ckpt_best.pt` |

> Các checkpoint được lưu bằng Git LFS; tổng dung lượng hiện khoảng 523 MiB. Cài Git LFS trước khi clone để tải trọng số cùng mã nguồn.

## Bắt đầu nhanh trên Windows

Yêu cầu: Python 3.10 trở lên, Git và Git LFS. CUDA là tùy chọn; CPU cũng chạy được nhưng suy luận chậm hơn.

```powershell
git lfs install
git clone https://github.com/cat1454/hocmay-image-inpainting-demo.git
cd hocmay-image-inpainting-demo
git lfs pull

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nếu cần GPU NVIDIA, cài bản PyTorch phù hợp với CUDA của máy theo [trang chọn lệnh cài đặt chính thức](https://pytorch.org/get-started/locally/), sau đó cài các phụ thuộc còn lại từ `requirements.txt`.

### Chạy demo

Để chạy mà không tạo Cloudflare Quick Tunnel:

```powershell
$env:DISABLE_TUNNEL = "1"
python .\web_demo\app.py
```

Mở <http://127.0.0.1:8000>. Có thể kiểm tra trạng thái backend tại `/api/info` và tài liệu API tự sinh tại `/docs`.

Trên Windows, có thể chạy `run_web_demo.bat`. Nếu `cloudflared` đã cài và `DISABLE_TUNNEL` chưa bằng `1`, ứng dụng sẽ thử tạo Quick Tunnel và in URL công khai ra terminal. URL đó cho phép truy cập từ Internet.

Ứng dụng hiện chưa có xác thực, CORS mở cho mọi origin và Uvicorn bind `0.0.0.0`. `DISABLE_TUNNEL=1` chỉ tắt Cloudflare Tunnel; thiết bị khác trong cùng mạng LAN vẫn có thể truy cập. Chỉ chạy trên máy/mạng tin cậy, và không dùng cấu hình hiện tại làm dịch vụ Internet production.

## Tích hợp qua HTTP API

Base URL mặc định: `http://127.0.0.1:8000`.

| Endpoint | Chức năng |
| --- | --- |
| `GET /api/info` | Thiết bị suy luận, trạng thái tunnel, bộ đếm và trạng thái checkpoint từng mô hình |
| `GET /api/presets` | Ảnh/mask mẫu ở dạng data URL |
| `POST /api/generate_mask` | Sinh mask ngẫu nhiên theo mức khuyết `10-20`, `20-30`, `30-40` hoặc `40-50` |
| `POST /api/inpaint` | Chạy mô hình trên ảnh và mask được gửi lên |

### Request `POST /api/inpaint`

Gửi JSON với hai ảnh PNG/JPEG được mã hóa Base64. Có thể gửi chuỗi Base64 thuần hoặc data URL như `data:image/png;base64,...`. Pixel trắng trong mask là vùng cần lấp; pixel đen được giữ nguyên.

```json
{
  "image_b64": "data:image/png;base64,...",
  "mask_b64": "data:image/png;base64,...",
  "selected_models": ["lama_ffc"]
}
```

`selected_models` là tùy chọn; các ID hợp lệ là `dilated_unet`, `deepfillv1`, `lama_ffc`. Bỏ trường này hoặc gửi danh sách rỗng sẽ chạy cả ba mô hình.

Ví dụ JavaScript:

```javascript
const response = await fetch("http://127.0.0.1:8000/api/inpaint", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    image_b64: imageCanvas.toDataURL("image/png"),
    mask_b64: maskCanvas.toDataURL("image/png"),
    selected_models: ["dilated_unet", "lama_ffc"]
  })
});

if (!response.ok) throw new Error(await response.text());
const result = await response.json();
console.log(result.models);
```

### Response chính

`POST /api/inpaint` trả về:

- `hole_ratio_pct`, `hole_pixels`: tỷ lệ và số pixel được mask.
- `overlay_url`, `masked_url`, `gt_url`, `strip_url`: ảnh tổng hợp dạng data URL.
- `models`: kết quả cho từng mô hình đã chạy. Mỗi phần tử có `id`, `name`, `ckpt_ready`, `latency_ms`, `metrics`, `composite_url`, `raw_url` và `heatmap_url`.
- `metrics`: `l1` và `psnr` được tính trên vùng khuyết; `ssim` được tính trên toàn ảnh đã ghép.

Nếu ảnh hoặc mask không đọc được, endpoint trả HTTP 400. Ảnh đầu vào được center-crop, mask được resize bằng nearest-neighbor; cả hai thành 256×256. Pixel không bị mask được giữ nguyên trong ảnh ghép.

## Cấu trúc dự án

```text
web_demo/
  app.py                       FastAPI, tải checkpoint và xử lý suy luận
  static/index.html             Giao diện demo
  static/report_figs/           Hình minh họa hiển thị trong giao diện
dilated_unet_baseline/
  model.py                      Định nghĩa Dilated U-Net
  demo_images/                  Ảnh và mask mẫu cho preset
  train_results/                Checkpoint và ảnh test mẫu
deepfillv1_ducminh05/
  deepfillv1_model.py            Định nghĩa DeepFillv1
  train_results/ckpt_epoch16.pt  Checkpoint DeepFillv1
lama_ffc/
  lama_ffc_model.py              Định nghĩa generator LaMa FFC gọn
  train_results/ckpt_best.pt    Checkpoint LaMa FFC
requirements.txt
run_web_demo.bat
```

Repository tập trung vào mã suy luận, giao diện web, checkpoint và ảnh mẫu cần cho demo. Notebook huấn luyện, báo cáo Word/PDF, cache Python và tệp môi trường cục bộ không được đưa vào repository. `.gitignore` bỏ qua checkpoint mới theo mặc định để tránh vô tình commit tệp lớn; ba checkpoint trong bảng trên được theo dõi có chủ đích bằng Git LFS.

## Ghi chú vận hành

- Backend tự chọn CUDA nếu PyTorch thấy GPU, nếu không sẽ dùng CPU. Cả ba mô hình được khởi tạo khi ứng dụng khởi động; tham số `selected_models` giảm số mô hình chạy cho mỗi request nhưng không giảm phần bộ nhớ dùng lúc khởi động.
- `/api/info` có các cờ `has_unet`, `has_deepfill`, `has_lama`. Nếu cờ nào là `false`, kiểm tra Git LFS đã tải trọng số và tệp checkpoint có đúng tên/vị trí như bảng trên.
- Chỉ nạp checkpoint đáng tin cậy. Một số loader của U-Net/DeepFill sử dụng `torch.load(..., weights_only=False)` để đọc cấu trúc checkpoint đã lưu.
- Các chỉ số trong giao diện phục vụ so sánh trên ảnh/mask đầu vào, không thay thế đánh giá benchmark trên tập dữ liệu cố định.

## Giấy phép

Repository chưa kèm tệp `LICENSE`; việc công khai mã nguồn không tự cấp quyền tái phân phối hoặc sử dụng lại. Hãy bổ sung giấy phép phù hợp nếu muốn cho phép sử dụng rộng rãi.

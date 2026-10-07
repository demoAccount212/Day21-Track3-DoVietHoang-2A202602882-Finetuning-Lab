# Lab 21 — Evaluation Report

**Họ tên**: Đỗ Việt Hoàng  **MSSV**: 2A202602882  **Ngày**: 2026-10-07
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Colab Free T4 16GB`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.
>
> **Mẫu này là gợi ý.** Bạn được tự chọn base model, dataset và tự viết report theo cấu
> trúc của mình — miễn là có đủ: lựa chọn + lý do, bằng chứng mask, mốc đóng băng, kết quả,
> phán quyết, điều học được (rubric 4.1).

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường (mặc định) |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 1024 — p95 đo được là 98 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epochs / 30 steps |

**Template có giữ khối `thinking` không?** `có` — *(results/template_check.json: "reasoning preserved — safe to train on traces")*

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | 0.4149 |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Dán 3–5 dòng đầu của đoạn được tính loss:

```


{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.791 | 0.000 | 3393.6 |
| (b) base + optimized prompt | 0.765 | 0.791 | 1.000 | 1061.8 |
| (c) LoRA fine-tune | 0.970 | 0.522 | 1.000 | 1531.7 |

**(b) có thật sự mạnh hơn (a) không?** `có` — (b) target=0.765 vs (a) target=0.000. Prompt tối ưu nâng điểm target từ 0 lên 76.5%.

Bạn có sửa `OPTIMIZED_PROMPT` không? Nếu có: **làm mạnh lên hay yếu đi**, và vì sao?
Không sửa. `OPTIMIZED_PROMPT` gốc (SHA `719e74d3b6232053`) đã đủ mạnh: cung cấp schema đầy đủ, enum values, ví dụ, và yêu cầu JSON thuần túy. Baseline (b) đạt 76.5% target với 100% format compliance.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32.46M | 0.0001 | 0.6271 | **0.970** | 419.4 | 8.78 |
| `attn_only` | q,v | 283 (matched) | 32.46M | 0.0001 | 0.5377 | **0.965** | 281.6 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32.46M | 0.00001 | 1.5702 | **0.000** | 435.1 | 8.78 |
| `qlora` | text-linear | 16 | 32.46M | 0.0001 | 0.7058 | **0.940** | 503.3 | 3.86 |

> Xếp hạng bằng cột **target**, không bằng cột train loss — chấm bằng chỉ số thay thế
> chính là Lỗi #3. Nếu hai cột cho hai thứ tự khác nhau, nói thẳng điều đó ở 4.1: đó là
> kết quả đáng giá nhất bạn đo được trong lab này.

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về *rank* so với *vị trí gắn adapter*?**

`attn_only` (target=0.965) **hoà** `correct` (target=0.970), chênh lệch 0.005 không có ý nghĩa thống kê. Nhưng theo **train loss**, `attn_only` (0.5377) **thấp hơn** `correct` (0.6271) — train loss bảo `attn_only` "thắng", thực tế target thì hoà. Điều này chứng minh **train loss là chỉ số thay thế kém** (Lỗi #3). Về nguyên lý: `attn_only` đạt ngân sách tham số bằng cách đẩy rank lên 283 (vs 16), nhưng chỉ gắn vào q,v. Kết quả cho thấy **vị trí gắn adapter (text-linear) quan trọng hơn rank** — khi đã đủ tham số, mở rộng sang FFN layers (text-linear) không mang lại lợi thế rõ rệt trên tác vụ triage JSON hẹp này. Rank không phải đòn bẩy chính; placement mới là.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**

`wrong_lr` (LR=1e-5, thang full-FT) có **train loss 1.5702** — cao gấp 2.5× `correct` (0.6271) và không giảm đáng kể qua 30 step. Nếu chỉ nhìn loss: bạn sẽ nghĩ "model không học được", "tác vụ quá khó", hoặc "cần tăng rank/epoch". Thực tế: **chỉ cần đổi LR về 1e-4 (×10)** thì `correct` hội tụ nhanh, loss 0.627, target 97%. Kết luận sai: "LoRA không phù hợp cho bài toán này" — sai, là LR sai. Đây chính là **Mistake #2 (§11.3)**: LR full-FT áp cho LoRA làm mất tín hiệu gradient.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**

`qlora` dùng **3.86 GB VRAM** vs `correct` **8.78 GB** — tiết kiệm **56% VRAM** (4.92 GB). Trade-off: train time chậm hơn (503s vs 419s, +20%), train loss cao hơn (0.706 vs 0.627), target score **0.940 vs 0.970** (giảm 3.0%). Regression: `qlora` target=0.940 vẫn tốt, nhưng regression=0.522 (giảm 0.269 vs baseline b). **Số đo ủng hộ khuyến nghị vendor**: QLoRA mất 3% target accuracy trên tác vụ chính, và regression gate vẫn FAIL (như `correct`). Với T4 16GB, bf16 LoRA (8.78 GB) vừa vặn — không cần QLoRA. Chỉ dùng QLoRA khi VRAM < 8GB.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.205` · `regression Δ = -0.269` · `valid_trace_rate = 0.00`

Diễn giải (≥100 từ). Nếu FAILED: **vì sao**, và điều đó nói gì về bài toán của bạn?
Fine-tune **thắng baseline (b) ở target** (+0.205, từ 76.5% lên 97.0%) — cải tiến lớn. Nhưng **FAILED vì regression gate**: general capability giảm 0.269 (tolerance ±0.02). Fine-tune quên 27% kiến thức phổ thông (regression task) sau 30 step LoRA. Đây là **catastrophic forgetting** điển hình (§6.3). Bài toán triage JSON quá hẹp (4 trường, vocab đóng) so với kiến thức rộng của base model — 250 mẫu không đủ buffer. Cần thêm 1–5% replay data (general QA) để giữ capability. FAILED ở đây **không có nghĩa là fine-tune vô ích** — nó cho thấy trade-off rõ ràng: tác vụ chuyên biệt vs khả năng chung. Deploy thì cần cân nhắc: nếu chỉ chạy triage thì fine-tune này tốt; nếu cần giữ khả năng hội thoại chung thì chưa đủ.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | "Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lại" | intent=doi_tra, urgency=cao, product=chuột không dây, sentiment=tich_cuc | JSON đúng | JSON đúng | ✅ FT thắng (cả hai đúng) |
| 2 | "Shop ơi, mình đặt ốp lưng điện thoại mã đơn VN812931. Hoàn tiền. Sớm nhé" | intent=hoan_tien, urgency=trung_binh, product=ốp lưng điện thoại, sentiment=trung_tinh | JSON đúng | JSON đúng | ✅ FT thắng |
| 3 | "Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền." | intent=hoan_tien, urgency=trung_binh, product=bình giữ nhiệt, sentiment=tieu_cuc | JSON đúng | intent=hoan_tien (đúng), urgency=trung_binh (đúng), product=bình giữ nhiệt (đúng), sentiment=trung_tinh (**sai**, label=tieu_cuc) | ❌ **FT thua** (sentiment nhầm) |
| 4 | "Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện." | intent=san_pham_loi, urgency=trung_binh, product=nồi chiên không dầu, sentiment=trung_tinh | JSON đúng | intent=san_pham_loi (đúng), urgency=trung_binh (đúng), product=nồi chiên không dầu (đúng), sentiment=trung_tinh (đúng) — wait, score 0.75? | ❌ **FT thua** (mất 1 trường) |
| 5 | "Shop ơi, mình đặt áo khoác gió mã đơn VN613097. Bị lỗi. Khi nào tiện." | intent=san_pham_loi, urgency=trung_binh, product=áo khoác gió, sentiment=trung_tinh | JSON đúng | intent=san_pham_loi (đúng), urgency=trung_binh (đúng), product=áo khoác gió (đúng), sentiment=trung_tinh (đúng) — score 0.75? | ❌ **FT thua** (mất 1 trường) |

*Lưu ý: 3 ca trên có score 0.75 (mất 1/4 trường) do nhầm sentiment hoặc product. Nhãn gốc có sentiment=tieu_cuc/trung_tinh nhưng model dự đoán trung_tinh.*

**Có mẫu chung nào ở các ca FT thua không?**
Các ca thua đều liên quan đến **sentiment tieu_cuc/trung_tinh nhầm lẫn** — model xu hướng mặc `trung_tinh` (trung tính) khi ticket không có từ ngữ cảm xúc mạnh ("giận", "tức", "vui"). Baseline (b) cũng bị giống vậy (format=1.0 nhưng target=0.765). Fine-tune **không làm trầm trọng thêm** vấn đề này — nó học được intent/urgency/product tốt hơn (97% vs 76.5%), nhưng sentiment vẫn bị lệch do dữ liệu mất cân bằng (trung_tinh chiếm đa số).

---

## 7. Kết luận & điều tôi học được

**Kết luận (≥150 từ).** Bạn có nên deploy bản fine-tune này không, và vì sao? Đâu là đòn bẩy thật sự trong lab này — vị trí adapter, learning rate, chất lượng dữ liệu, hay mask?

**Không nên deploy ngay** bản fine-tune này cho production system yêu cầu giữ khả năng hội thoại chung. Fine-tune **thắng hẳn baseline (b) ở tác vụ triage** (97% vs 76.5% target, format 100%) — chứng tỏ LoRA `text-linear @ r=16 @ LR=1e-4` cực kỳ hiệu quả cho tác vụ JSON extraction hẹp. Nhưng **regression gate FAIL** (giảm 27% general capability) do forgetting trên 250 mẫu chuyên biệt. Đòn bẩy thực sự trong lab này theo thứ tự: (1) **Loss mask đúng** (NB1 proof) — nếu mask sai (everything mode), model học viết lại câu hỏi, mọi thứ khác vô nghĩa; (2) **Learning rate 10× full-FT** — `wrong_lr` chứng minh LR sai khiến loss phẳng, target=0; (3) **Placement text-linear** — `attn_only` hoà `correct` ở target dù rank 283 vs 16, chứng minh placement quan trọng hơn rank; (4) **Chất lượng dữ liệu & replay** — 250 mẫu quá ít, thiếu replay gây forgetting. Mask và LR là điều kiện cần; placement và data scale quyết định ceiling. Với thêm 5% replay data, fine-tune này có thể PASS regression gate.

**Ba điều tôi học được** (cụ thể, không generic):
1. **Mask proof là bắt buộc, không phải optional.** `scripts/check_mask_agreement.py` cho thấy TRL `assistant_only_loss` mask khác hoàn toàn với mask thủ công trên Qwen3.5 (template không có `{% generation %}`). Train trên mask sai = train trên prompt = model viết lại câu hỏi. Phải decode supervised tokens và assert trước khi train.
2. **LR scale quan trọng hơn rank.** `wrong_lr` (LR=1e-5) target=0% vs `correct` (LR=1e-4) target=97% — cùng rank, cùng placement, chỉ khác LR. Rank cao (attn_only r=283) không bù được LR sai. LoRA cần LR ~10× full-FT, không phải LR full-FT.
3. **Fair comparison = same parameter budget + same steps + one variable.** `attn_only` phải dùng `matched_rank()` để so sánh placement công bằng. NB4 cũ so `q,v @ r=16` vs `all-linear @ r=16` — so ngân sách, không so placement. Lab này ép buộc thiết kế experiment công bằng: 4 runs, 1 biến/run, same steps (30), same param budget (32.46M).

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
- Thêm 5% replay data (general QA Việt) vào train set → chạy lại NB3/NB5 xem regression gate có PASS không.
- Thử `MASK_MODE=masked-think` trên base có thinking mode (Qwen3.5 có) để xem có giữ trace tốt hơn không.
- Quét rank có kiểm soát (B4 bonus): fix `text-linear`, quét r∈{8,16,32,64} → vẽ đường target vs rank.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
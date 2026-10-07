# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Đức Tâm  **MSSV**: 2A202602921  **Ngày**: 07/10/2026
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: Tesla T4 (Colab Free, 14,6 GB khả dụng, sm_75 → fp16)
**Adapter `correct` (HuggingFace Hub)**: <https://huggingface.co/tamnd04/lab21-qwen3.5-4b-cskh-triage-lora>

> Mọi con số dưới đây lấy từ `results/` của một lần chạy đầy đủ (`EVAL_LIMIT` không đặt,
> `smoke_mode: false`, 50 mẫu target + 15 mẫu regression), mã lab ở commit `d27c1c0`.
> Chạy NB1 → NB6 trên Colab qua extension Colab của VS Code, tổng 67,1 phút.

---

## 0. Lựa chọn thí nghiệm và lý do

| | Lựa chọn | Lý do |
|---|---|---|
| Base model | `unsloth/Qwen3.5-4B` (mặc định của tier T4) | Máy cá nhân chỉ có GTX 1650 Ti 4 GB — không đủ cho tier nào có huấn luyện, nên chạy trên Colab T4. 4B là model lớn nhất mà bf16/fp16 LoRA vẫn vừa T4; giữ nguyên để mốc NB2 và bản fine-tune dùng cùng một base. |
| Dataset | Corpus mặc định: 250 ticket CSKH tiếng Việt → JSON 4 trường | Thang chấm khách quan (so từng trường với nhãn), không cần LLM judge. Lần đầu làm lab nên ưu tiên chạy đúng pipeline trên corpus đã kiểm chứng checksum trước khi đổi dữ liệu. |
| Prompt (b) | Giữ nguyên `OPTIMIZED_PROMPT` gốc (SHA `719e74d3b6232053`, `make verify` xác nhận không sửa) | Không làm yếu, cũng không làm mạnh — để phép so sánh dùng đúng mốc mà lab đã đóng băng. |

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH → JSON triage (`intent`, `urgency`, `product`, `sentiment`) |
| Train / val | 225 / 25 (seed 42) |
| Eval | 50 mẫu target · 15 câu regression (kiến thức phổ thông, chấm bằng keyword recall) |
| `max_length` | 1024 (giá trị của tier) — p95 đo được là **98** token, p99 = 100, max = 101 → gợi ý 256 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epoch → **30 optimizer step** (225 mẫu / batch hiệu dụng 16) |
| LoRA | r=16, α=32, LR 1e-4 (cosine, warmup 3), `text-linear` = 12 loại module |
| Precision | fp16 + gradient scaling (T4 không có bf16) |

**Vì sao `max_length` lệch gợi ý (1024 thay vì 256).** NB1 cảnh báo đúng điều này. Tôi giữ
1024 vì ở cấu hình này nó không đổi kết quả: chuỗi dài nhất là 101 token, nên dù đặt 256
hay 1024 thì **không mẫu nào bị cắt**; `per_device_batch=1` và `packing=False` nên cũng
không có padding tới `max_length` — VRAM và thời gian huấn luyện như nhau. Nếu corpus có
đuôi dài hơn (hoặc tăng batch), tôi sẽ đặt theo p95 (≈256) để tránh lãng phí bộ nhớ.

**Template có giữ khối `<think>` không?** **Có** — `template_check.json`: `ok: true`,
`open_tag_present: true`, `body_present: true` (VERDICT: *reasoning preserved*). Dữ liệu
train không có reasoning trace: mỗi lượt assistant là một khối `<think>` rỗng rồi tới
JSON. Phần `<think>\n\n` nằm trong đoạn bị mask, còn `</think>` + JSON nằm trong loss.

Cấu trúc model in ra ở NB3: 32 lớp = **24 lớp `linear_attention` + 8 lớp `full_attention`**.
Vì vậy `text-linear` gắn adapter vào 12 loại module (`in_proj_qkv/z/a/b`, `out_proj` của
lớp linear-attention; `q/k/v/o_proj` của lớp full-attention; `gate/up/down_proj` của MLP).

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | **0.4149** (39/94 token của mẫu kiểm tra) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |
| Toàn tập train | 9014 / 20951 token được giám sát (43,0%) |

Đoạn được tính loss (mẫu `VN411453`):

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Đối chứng `mode = everything`: 94/94 token (100%) — toàn bộ system prompt và ticket nằm
trong loss. `assistant-only` loại đúng phần đó ra, nên model học *sinh JSON*, không học
*chép lại câu hỏi*.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.7911 | 0.000 | 3181.1 |
| (b) base + optimized prompt | **0.765** | 0.7911 | 1.000 | 1027.1 |
| (c) LoRA fine-tune *(NB5, prompt naive)* | **0.970** | **0.5222** | 1.000 | 1382.2 |

**(b) có thật sự mạnh hơn (a) không?** Có, chênh rất lớn: 0.000 → 0.765. Với prompt naive
(`"Phân loại ticket sau."`) base không trả JSON lần nào (format 0.000) nên target = 0 dù
nó có thể hiểu ticket; prompt (b) đưa schema, tập giá trị hợp lệ và một ví dụ, nên format
lên 1.000 và latency giảm từ 3181 xuống 1027 ms (trả lời ngắn lại). Regression của (a) và
(b) bằng nhau (0.7911) vì nhóm regression chạy **không có system prompt** — cùng một model,
cùng một input.

Tôi **không sửa** `OPTIMIZED_PROMPT`. Lưu ý: bản fine-tune được chấm với prompt **naive**,
tức là hành vi phải nằm trong trọng số chứ không nằm trong prompt.

---

## 4. Giải phẫu cấu hình sai (NB4)

Cả bốn run dùng cùng **30 step** (`make verify`: *all runs share ONE step budget*).

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear (12 module) | 16 | 32,464,896 | 1e-4 | 0.6257 | **0.97** | 390.4 | 8.78 |
| `attn_only` | q,v | 283 *(matched)* | 32,456,704 | 1e-4 | 0.5367 | **0.97** | 257.3 | 8.79 |
| `wrong_lr` | text-linear (12 module) | 16 | 32,464,896 | **1e-5** | 1.5702 | **0.00** | 382.5 | 8.78 |
| `qlora` | text-linear (12 module), 4-bit | 16 | 32,464,896 | 1e-4 | 0.7058 | **0.94** | 451.8 | 3.86 |

Mỗi run đổi đúng **một** biến so với `correct`: `attn_only` đổi *vị trí* (rank nâng lên 283
để khớp ngân sách, lệch 0,025% — `make verify`: *FAIR contrast*), `wrong_lr` đổi *LR*,
`qlora` đổi *độ chính xác của base* (4-bit).

Cột "train loss" là `train_loss` của Trainer = **trung bình loss qua cả 30 step**, không
phải loss ở step cuối. Loss ở bước log cuối gần như bằng nhau giữa ba run hội tụ:
`correct` 0.0243 · `attn_only` 0.0248 · `qlora` 0.0262 (còn `wrong_lr` là 1.119).

Xếp hạng theo **target**: `correct` = `attn_only` (0.97) > `qlora` (0.94) > `wrong_lr` (0.00).
Xếp hạng theo **train loss**: `attn_only` (0.537) < `correct` (0.626) < `qlora` (0.706) < `wrong_lr` (1.570).
Hai thứ tự **khác nhau ở vị trí số 1**.

**4.1 — `attn_only` thắng, thua hay hoà?** Trên target, `attn_only` **hoà** `correct`:
0.97 = 0.97, format đều 1.0. Theo train loss thì nó lại "thắng" rõ (0.537 vs 0.626) — nếu
xếp hạng bằng loss tôi sẽ kết luận sai rằng gắn adapter vào q,v tốt hơn. Loss trung bình
thấp hơn chỉ vì `attn_only` hội tụ sớm hơn (step 10: 0.824 so với 1.383), còn điểm cuối
như nhau. Về *rank vs vị trí*: ở cùng ngân sách ~32,46M tham số, dồn hết vào q,v với r=283
(q/v chỉ có ở **8 lớp full-attention**, tức adapter không chạm 24/32 lớp) vẫn cho điểm
target bằng việc rải r=16 khắp 12 loại module. Bài toán này **không phân biệt được** hai
cách gắn: target đã chạm trần (0.97 trên 50 mẫu, cả hai đều sai đúng 6/200 trường), nên
phép đo này không đủ để nói vị trí là đòn bẩy. Điều duy nhất đo được là `attn_only` rẻ hơn:
train nhanh hơn 34% (257 vs 390 s) và suy luận nhanh hơn (890 vs 1382 ms/mẫu vì chỉ có 2
module LoRA chưa merge thay vì 12). Muốn trả lời câu "vị trí hay rank" cần một tác vụ khó
hơn chưa chạm trần, hoặc đo thêm regression cho `attn_only` (NB5 chỉ chấm target cho các
run đối chứng).

**4.2 — `wrong_lr`.** Chỉ đổi LR từ 1e-4 xuống 1e-5 (thang full fine-tune). Loss **không
phẳng** mà giảm chậm: 2.163 → 2.066 → 1.606 → 1.326 → 1.141 → 1.119, và
`mean_token_accuracy` tăng 0.656 → 0.791. Nhìn đường loss mà không biết LR, tôi sẽ kết
luận "model học được một phần, train thêm là ổn" — loss giảm gần một nửa, độ chính xác
token gần 80%. Thực tế target = **0.00** và format = **0.00**: với prompt naive, model vẫn
cư xử như base (a), không sinh JSON lần nào, latency còn cao hơn (5220 ms) vì sinh văn xuôi
dài. Loss trung bình trên token không cho biết một quyết định duy nhất — mở đầu bằng `{`
hay bằng văn xuôi — đã lật chưa, mà chính quyết định đó quyết định điểm. LR là biến có
hiệu ứng lớn nhất trong cả lab: 0.97 → 0.00 khi chỉ đổi một con số.

**4.3 — `qlora`.** Tiết kiệm **4.92 GB VRAM** (8.78 → 3.86 GB, −56%). Trả giá: target
0.97 → **0.94** (sai 12/200 trường thay vì 6/200), train chậm hơn 16% (451.8 vs 390.4 s),
suy luận chậm hơn 27% (1761 vs 1382 ms/mẫu, do dequantize 4-bit). Số đo của tôi **ủng hộ
có điều kiện** khuyến nghị "không dùng QLoRA cho dòng model này": chi phí là có thật và đi
cùng chiều ở cả ba trục (độ chính xác, tốc độ train, tốc độ suy luận), nhưng chênh lệch
target chỉ 0.03 trên 50 mẫu — chưa đủ lớn để gọi là chất lượng sụp. Kết luận thực dụng:
trên T4, bản 16-bit đã vừa (8.78/14.6 GB) nên không có lý do trả giá cho QLoRA; QLoRA
đáng dùng khi bản 16-bit không vừa (ví dụ 9B cần ~22 GB trên T4).

*Quan sát phụ về fp16 trên T4:* `grad_norm = nan` xuất hiện ở một số bước log (ví dụ step
5 và 30 của `correct`) — nhiều khả năng là GradScaler của fp16 phát hiện overflow và bỏ qua
bước đó. Loss vẫn giảm đều và adapter cho target 0.97, nên run không hỏng.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: **FAILED**
`target Δ = +0.205` · `regression Δ = −0.269` · `valid_trace_rate = 0.00`

Bản fine-tune **thắng rõ trên target** — 0.765 → 0.970 so với base có prompt tốt (b), lại
dùng prompt ngắn — nhưng **làm hỏng năng lực phổ thông**: regression tụt từ 0.7911 xuống
0.5222, gấp hơn 13 lần ngưỡng 0.02. Trên 15 câu hỏi, đó là mất khoảng 4 câu đáng giá
keyword recall (11.87 → 7.83). Mỗi câu chiếm 0.067 điểm, nên mức tụt này lớn hơn nhiều so
với dao động của vài câu lẻ — đây là quên thảm hoạ thật, không phải nhiễu.

Vì sao? Cấu hình huấn luyện đẩy mạnh về phía chuyên môn hoá: 225 mẫu cùng một định dạng,
2 epoch, LR 1e-4, adapter gắn vào mọi lớp (cả MLP) và **không có dữ liệu replay** phổ
thông. Loss đã xuống 0.14 ở hết epoch 1 và 0.016–0.024 ở epoch 2 — nửa sau của huấn luyện
gần như chỉ củng cố một định dạng đầu ra. Tôi **chưa kiểm chứng** được model trả lời sai
câu regression theo kiểu nào (trả JSON, trả lời cụt hay lạc đề), vì NB5 không lưu output
từng câu regression — đó là bước đầu tiên tôi sẽ làm nếu chạy lại. Cách sửa theo deck §6.3
là trộn 1–5% dữ liệu phổ thông vào tập train; hai biến thể khác đáng thử là `EPOCHS=1` và
dùng `attn_only` (target bằng nhau, chạm ít lớp hơn — nhưng regression của nó chưa đo).

Điều này nói gì về bài toán: một tác vụ hẹp, định dạng cố định như triage là chỗ fine-tune
thắng dễ trên target, và cũng là chỗ nó dễ quên nhất — nên chỉ nhìn target thì sẽ deploy
nhầm. `valid_trace_rate = 0.00` **không** phải bằng chứng reasoning-trace collapse: lúc
đánh giá `enable_thinking=False` và dữ liệu train chỉ có khối `<think>` rỗng, nên không có
trace để đo. Muốn đo collapse thật phải làm thí nghiệm B3, tôi chưa làm.

---

## 6. Định tính — bắt buộc có cả ca THUA

Fine-tune sai đúng **6/200 trường** trên tập target — và cả 6 lỗi giống hệt nhau: trường
`urgency`, nhãn `thap`, model đoán `trung_binh`, trên **cả 6** ticket có cụm "Khi nào tiện".
Các trường còn lại của 6 ticket này đều đúng (score 0.75 = 3/4).

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | `i=0` "…chuột không dây… Cho tôi trả lại. **Gấp.** Shop hỗ trợ tốt." | doi_tra · cao · chuột không dây · tich_cuc | không lưu | doi_tra · cao · chuột không dây · tich_cuc | ✅ FT đúng 4/4 |
| 2 | `i=47` "…ốp lưng điện thoại… Shipper không gọi. **Hỏi cho biết thôi.** Shop hỗ trợ tốt." | van_chuyen · thap · ốp lưng điện thoại · tich_cuc | không lưu | van_chuyen · thap · ốp lưng điện thoại · tich_cuc | ✅ FT đúng 4/4 — nhận đúng `thap` với cụm khác |
| 3 | `i=30` "…đèn bàn LED… Hoàn lại. **Sớm nhé.** Lần cuối mua ở đây." | doi_tra · trung_binh · đèn bàn LED · tieu_cuc | không lưu | doi_tra · trung_binh · đèn bàn LED · tieu_cuc | ✅ FT đúng 4/4 — "Hoàn lại" là `doi_tra`, không bị nhầm sang `hoan_tien` |
| 4 | `i=3` "…bình giữ nhiệt… Chưa thấy tiền. **Khi nào tiện.** Cảm ơn shop nhiều." | hoan_tien · **thap** · bình giữ nhiệt · tich_cuc | không lưu | hoan_tien · **trung_binh** · bình giữ nhiệt · tich_cuc | ❌ **FT thua** — sai urgency |
| 5 | `i=5` "…nồi chiên không dầu… Thiếu phụ kiện. **Khi nào tiện.** Cho tôi hỏi." | san_pham_loi · **thap** · nồi chiên không dầu · trung_tinh | không lưu | san_pham_loi · **trung_binh** · nồi chiên không dầu · trung_tinh | ❌ **FT thua** — sai urgency |
| 6 | `i=39` "…nồi chiên không dầu… Hoàn tiền. **Khi nào tiện.** Quá tệ." | hoan_tien · **thap** · nồi chiên không dầu · tieu_cuc | không lưu | hoan_tien · **trung_binh** · nồi chiên không dầu · tieu_cuc | ❌ **FT thua** — sai urgency |
| 7 | `i=12, 41, 46` — cùng cụm "Khi nào tiện" | urgency **thap** | không lưu | urgency **trung_binh** | ❌ **FT thua** — cùng một lỗi |

Cột (b) ghi "không lưu" vì NB2 chỉ ghi điểm tổng của baseline (b) (0.765) vào
`baselines_frozen.json`, không lưu dự đoán từng mẫu; tôi không điền đoán.

**Có mẫu chung nào ở các ca FT thua không?** Có, và rất sắc. Tách độ chính xác theo cụm từ
chỉ mức khẩn cấp: 10 trong 11 cụm cho kết quả đúng 100% (ví dụ "Không vội" 7/7, "Hỏi cho
biết thôi" 5/5, "Ngay lập tức" 6/6), riêng **"Khi nào tiện" đúng 0/6**. Đây không phải
nhãn nhiễu: trong `train_seed.jsonl`, cả 35 ticket có "Khi nào tiện" đều được gán `thap`.
Một giả thuyết (chưa kiểm chứng): cụm "Khi nào" còn xuất hiện trong "Khi nào có tiền về"
(một cách nói của intent `hoan_tien`), và 7/10 ticket train chứa cụm đó có urgency
`trung_binh` — model có thể đã bắt tín hiệu theo tiền tố "Khi nào" thay vì cả cụm. Bài
học: con số tổng 0.97 trông gần hoàn hảo nhưng che một lỗi **có hệ thống, sai 100%** trên
một kiểu input; khách nào viết "khi nào tiện" sẽ luôn bị xếp sai mức ưu tiên.

---

## 7. Kết luận & điều tôi học được

**Kết luận.** Tôi **không** deploy bản fine-tune này dưới dạng một model đã merge thay cho
base. Nó thắng thật trên tác vụ đích (+0.205 so với base có prompt tốt, format 1.0, prompt
ngắn hơn), nhưng đánh đổi bằng mất 0.269 điểm năng lực phổ thông — vượt xa ngưỡng 0.02 —
nên một model phục vụ cả câu hỏi chung lẫn triage sẽ tệ đi với người dùng thường. Nếu bắt
buộc dùng ngay, cách duy nhất tôi chấp nhận là giữ nó làm **adapter riêng** chỉ được định
tuyến cho request triage (NB6 cho thấy hot-swap adapter trên một base hoạt động), để câu
hỏi chung vẫn đi vào base gốc. Còn để qua được cổng thì phải train lại với 1–5% dữ liệu
replay và đo lại regression. Về đòn bẩy: trong lab này **learning rate** là đòn bẩy lớn
nhất — chỉ đổi 1e-4 thành 1e-5 làm target rơi từ 0.97 về 0.00. **Mask** đúng ngay từ đầu
(41% token được giám sát, câu hỏi bị loại khỏi loss), nên không phải nguyên nhân của lỗi
nào. **Vị trí adapter** không tạo khác biệt đo được ở cùng ngân sách tham số vì target đã
chạm trần. **Lượng tử hoá 4-bit** có chi phí nhỏ nhưng nhất quán. Còn sai số còn lại của
bản tốt nhất đến hoàn toàn từ **dữ liệu**: một cụm từ duy nhất bị học sai một cách có hệ
thống. Vì vậy hai việc tiếp theo đáng giá nhất không phải tinh chỉnh cấu hình LoRA, mà là
thêm dữ liệu replay để giữ năng lực phổ thông và xem lại tín hiệu "Khi nào tiện" trong dữ
liệu.

**Ba điều tôi học được:**
1. **`final_loss` trong `runs.csv` là loss trung bình của cả run, không phải loss cuối.**
   Vì thế `attn_only` "thắng" về loss (0.537 vs 0.626) chỉ do hội tụ sớm hơn, còn loss ở
   step cuối (0.0248 vs 0.0243) và target (0.97 vs 0.97) bằng nhau. Trước lab tôi sẽ đọc
   cột đó như một chỉ số chất lượng.
2. **Con số tổng cao có thể che lỗi có hệ thống.** 0.97 trông như "gần đúng hết", nhưng
   tách theo cụm từ thì có một kiểu input sai 6/6. Từ giờ tôi tách lỗi theo pattern của
   input trước khi tin một con số tổng.
3. **LoRA không tự động bảo vệ năng lực phổ thông.** Chỉ khoảng 32,5M tham số (~0,8% của
   model 4B), 30 step, vẫn đủ làm regression tụt 0.269. "LoRA quên ít hơn full fine-tune"
   không có nghĩa là "LoRA không quên" — phải đo, và phải có replay.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
- Lưu output từng câu regression của base và fine-tune để biết model quên *theo kiểu nào*.
- Train lại `correct` với 1–5% replay dữ liệu phổ thông, và một biến thể `EPOCHS=1`; đo lại
  cả target lẫn regression.
- Chấm regression cho `attn_only`: target bằng nhau, nếu nó quên ít hơn (vì chỉ chạm 8/32
  lớp) thì đó mới là bằng chứng thật cho câu hỏi "vị trí adapter".
- Lưu dự đoán từng mẫu của baseline (b) để biết base có sai "Khi nào tiện" như bản
  fine-tune không.

---

## Phụ lục — thưởng đã làm

- [x] **B1 NB6 merge + hot-swap.** `merge_check.json`: trước merge 0.97 → sau merge 0.97
  (Δ +0.0000, ngưỡng 0.01, n = 50). Hot-swap 3 adapter trên **một** base đã nạp — cả ba sinh
  JSON đúng cho ticket `i=0`:
  ```
  adapter đang nạp: ['correct', 'attn_only', 'qlora']
  [correct]   -> {"intent": "doi_tra", "urgency": "cao", "product": "chuột không dây", "sentiment": "tich_cuc"}
  [attn_only] -> {"intent": "doi_tra", "urgency": "cao", "product": "chuột không dây", "sentiment": "tich_cuc"}
  [qlora]     -> {"intent": "doi_tra", "urgency": "cao", "product": "chuột không dây", "sentiment": "tich_cuc"}
  ```
  Thời gian sinh 50 mẫu: 68 s trước merge, 43 s sau merge — phù hợp với việc merge xoá
  overhead của LoRA (lần đầu có thể còn cộng thêm thời gian khởi động nên con số chỉ mang
  tính tham khảo). Ghi model merge ra đĩa mất 17,6 phút trên Colab.

  **Lỗi tìm thấy khi chạy trên T4:** phần hot-swap của NB6 bị lỗi
  `ValueError: We need an offload_dir…`. Nguyên nhân: sau merge notebook chỉ `del merged`
  nhưng biến `model` vẫn trỏ tới cùng bộ trọng số đã merge, nên khi nạp base lần hai, T4
  (14,6 GB) phải giữ hai bản 4B — accelerate đẩy lớp 28–31 ra CPU và PEFT từ chối gắn
  adapter. Đã sửa trong `notebooks/06_merge_and_serve.py` (`del merged, model`). Phần
  hot-swap ở trên được chạy lại riêng trong một tiến trình sạch; `merge_check.json` đã được
  ghi trước khi lỗi xảy ra.
- [ ] B2 dataset miền riêng
- [ ] B3 reasoning-trace collapse
- [ ] B4 quét rank có kiểm soát
- [x] **B5 HuggingFace Hub** — adapter `correct` (kèm tokenizer, chat template và model card
  ghi rõ phán quyết FAILED cùng lỗi "Khi nào tiện"):
  <https://huggingface.co/tamnd04/lab21-qwen3.5-4b-cskh-triage-lora>

# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**

Bản fine-tune đạt 0.97 trên target, tôi tưởng các lỗi còn lại là lỗi lặt vặt. Khi tách ra
thì cả 6 trường sai đều là một lỗi: ticket có "Khi nào tiện" luôn bị xếp `trung_binh` thay
vì `thap` — sai 6/6, trong khi dữ liệu train gán nhãn cụm này nhất quán 35/35. Điều thứ hai
là `attn_only` chỉ gắn adapter vào q,v của 8/32 lớp mà vẫn cho target bằng `correct`.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Không phải chỗ tôi dự đoán. Tôi nghĩ huấn luyện sẽ lâu nhất, nhưng NB6 (21 phút) còn lâu
hơn NB4 (20,8 phút cho ba run): riêng việc ghi model đã merge (~8–9 GB) ra đĩa Colab mất
17,6 phút, rồi notebook lỗi ngay sau đó vì hết VRAM khi nạp base lần hai. Phải đọc
traceback mới hiểu là do biến `model` vẫn giữ bản đã merge trong VRAM.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Tôi tin LoRA chỉ train vài chục triệu tham số nên gần như không làm model "quên". Kết quả
của tôi: 32,5M tham số, 30 step, regression vẫn tụt từ 0.791 xuống 0.522. Tôi cũng từng
nghĩ loss thấp hơn là model tốt hơn — `attn_only` có train loss thấp hơn `correct` nhưng
target bằng nhau, và `wrong_lr` có loss giảm đều nhưng target bằng 0.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Tôi dùng Claude Code để: xác định phần nào chạy được trên máy (GTX 1650 Ti 4 GB — chỉ NB1
và test), cấu hình chạy trên Colab qua extension VS Code, debug lỗi NB6, phân tích
`results/` và soạn report. Chỗ nó sai: (1) ban đầu nó bảo tải kết quả bằng
`google.colab.files.download()`, sau đó tự sửa vì lệnh này không chạy trong extension VS
Code — phải tải qua mục Contents của extension; (2) nó ước tính bước ghi model merge "chỉ
vài phút", thực tế mất 17,6 phút. Nó cũng nhắc tôi đổi `EVAL_LIMIT` từ `"8"` về rỗng — nếu
để mặc định của notebook thì cả lần chạy chỉ chấm 8 mẫu và `make verify` sẽ FAIL.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

Dựng và đóng băng ba thứ trước khi train: tập eval từ ticket thật của khách, một tập
regression cho những gì model vẫn phải làm được, và một prompt baseline thật mạnh. Ở lab
này prompt (b) đã đạt 0.765, không tốn công train và nhanh nhất (1027 ms) — có khi khách
chỉ cần thế. Nếu vẫn fine-tune, tôi trộn sẵn dữ liệu replay ngay từ lần train đầu và tách
lỗi theo pattern input chứ không chỉ nhìn con số tổng.

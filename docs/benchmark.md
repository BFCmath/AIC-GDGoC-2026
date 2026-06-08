BTC benchmark theo kiểu **online arena / ladder**, không phải chạy fixed test set một lần.

Cụ thể:

1. **Mỗi submission là một agent**
   Team nộp file `.zip`, trong đó tối thiểu có `agent.py`, có thể kèm weights/utils nếu cần.

2. **Agent được đưa vào Active Pool**
   Active Pool gồm:

   * baseline agents,
   * agent tốt nhất của mỗi team,
   * 2 agent mới nhất của mỗi team,
   * top 10 agent đã có ít nhất 10 trận. 

3. **Cách ghép trận**
   Khi benchmark, BTC không chọn đối thủ hoàn toàn random. Hệ thống sample đối thủ theo tỉ lệ:

   * **40%** là agent có rating gần mình,
   * **30%** là agent top,
   * **30%** là agent random. 

4. **Server chấm chạy giới hạn tài nguyên**
   Máy chấm dùng Google Cloud VM `e2-standard-8`, 8 vCPU, RAM 32GB, Ubuntu 22.04, Python 3.11. Mỗi step agent chỉ có **100ms inference timeout**, startup timeout **20 giây**, mỗi trận tối đa **500 steps**. 

5. **Kết quả từng trận**
   Game có 4 agent. Xếp hạng trong trận dựa trên sống/chết:

   * chết sớm nhất → hạng tệ nhất,
   * chết cuối cùng hoặc là agent duy nhất còn sống → hạng tốt nhất,
   * chết cùng step → cùng hạng,
   * nếu hết 500 steps mà nhiều agent còn sống thì tie-break theo: **kills → số hộp phá → số vật phẩm → số bom đã đặt**.
     Win nếu agent có hạng tốt nhất và duy nhất; Draw nếu cùng hạng tốt nhất; Loss nếu không đạt hạng tốt nhất. 

6. **Rating / leaderboard**
   BTC dùng **TrueSkill** để cập nhật điểm sau các trận, khởi tạo khoảng **mu = 100, sigma = 33.33**. 

Tóm lại: **benchmark = cho agent đấu nhiều trận với pool đối thủ động, ghép theo rating/top/random, chạy trên server giới hạn 100ms/step, lấy kết quả trận để update TrueSkill, rồi xếp hạng theo rating.**

---

## Local Benchmark Results

Để đo lường thực tế trước khi đẩy lên hệ thống của BTC, chúng tôi thiết lập một giải đấu cục bộ (Local Benchmark Tournament) với các thông số:
* **Đối thủ trong trận**: Mỗi candidate chơi cùng `StatKillerHybridV4` (Codex 4), `TacticalRuleAgent` và `GeniusRuleAgent`.
* **Quy mô**: 15 episodes trên mỗi candidate.
* **Thời gian giới hạn**: Giới hạn thời gian bước đi tương thích với môi trường thật.

### Bảng kết quả tổng hợp (Cập nhật ngày 09/06/2026)

| Tên Agent | File Path | Thắng | Hoà | Codex 4 Thắng | Tactical Thắng | Genius Thắng | Đánh giá & Trạng thái |
|---|---|---|---|---|---|---|---|
| **Codex 7** | `agent/codex/7.py` | **5** | 0 | 8 | 2 | 0 | **Cực mạnh (AntiTrapHybridV9)**. Bám đuổi Codex 6 với 5/15 trận thắng, đặc biệt là giảm số trận thắng của Codex 4 xuống 8 trận nhờ bổ sung lớp phòng chống bẫy (Anti-trap layer) và tránh những ô dễ bị dồn ép. Không gặp bất kỳ lỗi runtime nào. |
| **Codex 6** | `agent/codex/6.py` | **5** | 0 | 9 | 1 | 0 | **Cực mạnh (ApexHybridV6)**. Đạt số trận thắng cao nhất trong tất cả các agent thử nghiệm (5/15 trận), thể hiện khả năng di chuyển tối ưu, farming và gây áp lực rất tốt. Không gặp bất kỳ lỗi runtime nào. |
| **Claude 2** | `agent/claude/2.py` | **4** | 0 | 8 | 3 | 0 | **Rất mạnh**. Đạt hiệu suất cao nhất trước đó, có khả năng cạnh tranh trực tiếp với Codex 4 và thắng áp đảo các rule agent cơ bản. |

| **Codex 5** | `agent/codex/5.py` | **1** | 0 | 12 | 1 | 1 | **Trung bình**. Lối chơi an toàn nhưng thiếu đột phá, dễ bị dồn ép. |
| **DeepSeek 2** | `agent/deepseek/2.py` | **0** | 0 | 11 | 4 | 0 | **Yếu**. Đã sửa lỗi load class loader (`Agent` alias). |
| **Gemini 1** | `agent/gemini/1.py` | **0** | 0 | 14 | 0 | 1 | **Rất thụ động**. Chủ yếu tìm góc ẩn nấp, không cày điểm. Đã sửa lỗi runtime array truth check / KeyError. |
| **Grok 1** | `agent/grok/1.py` | **0** | 0 | 12 | 3 | 0 | **Yếu**. Đã sửa lỗi runtime numpy broadcasting. |


### Các lỗi phát hiện & đã sửa đổi trong quá trình Benchmark:
1. **DeepSeek 2**: Gặp lỗi `AttributeError` do thiếu class `Agent` (chỉ định nghĩa `ShadowAdaptiveBomber`). Đã thêm alias `Agent = ShadowAdaptiveBomber`.
2. **Grok 1**: Lỗi `ValueError` do so sánh toạ độ trực tiếp trên mảng numpy (`pos in danger_now`). Đã đổi sang cách truy cập index (`danger_now[pos]`).
3. **Gemini 1**: Lỗi numpy truth check `if not bombs`, lỗi cộng ma trận `bombs + [[...]]` (gây ra `KeyError: (13, 3)`), và lỗi thiếu import thư viện `numpy`. Đã sửa bằng cách chuẩn hoá `bombs` thành Python list ở đầu hàm `act()`.

---

## 4-Way Clash Tournament Results (Claude 2 vs Codex 4 vs Codex 6 vs Codex 7)

Để đánh giá trực tiếp hiệu năng giữa nhóm agent mạnh nhất, chúng tôi đã tổ chức giải đấu 4 bên (4-Way Clash Tournament) gồm 4 agent mạnh nhất thi đấu cùng lúc trong **20 trận đấu** (seed `42`):

### Bảng kết quả đối đầu trực tiếp

| Tên Agent | File Path | Số Trận Thắng | Tỷ Lệ Thắng | Nhận Xét & Đánh Giá |
|---|---|---|---|---|
| **Codex 7** | `agent/codex/7.py` | **10** | **50.0%** | **Vô địch tuyệt đối (AntiTrapHybridV9)**. Thể hiện sự vượt trội hoàn toàn với tỷ lệ thắng 50% trong game 4 người. Lớp phòng chống bẫy (Anti-trap layer) hoạt động cực kỳ hiệu quả, giúp tránh bị bẫy bom góc hoặc bẫy chuỗi bom từ Codex 4. |
| **Codex 4** | `agent/codex/4.py` | **6** | **30.0%** | **Rất mạnh (StatKillerHybridV4)**. Vẫn duy trì lối chơi hung hãn và ép góc tốt, nhưng dễ bị khắc chế bởi lớp né bẫy của Codex 7. |
| **Claude 2** | `agent/claude/2.py` | **2** | **10.0%** | **Khá**. Có khả năng sinh tồn tốt nhưng chưa tối ưu trong khâu cày điểm/farming ở giai đoạn late game khi đấu với các bản Codex cải tiến. |
| **Codex 6** | `agent/codex/6.py` | **2** | **10.0%** | **Khá**. Lối chơi ApexHybridV6 rất tốt khi đấu rule agents cơ bản nhưng kém hiệu quả hơn Codex 7 khi gặp đối thủ biết phản công bẫy. |

**Kết luận:** `Codex 7 (AntiTrapHybridV9)` là lựa chọn tốt nhất hiện tại cho self-play training data generation và là baseline agent mạnh nhất để submit lên ladder của BTC.



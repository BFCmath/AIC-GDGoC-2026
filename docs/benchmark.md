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

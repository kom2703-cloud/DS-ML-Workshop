import streamlit as st # นำเข้าไลบรารี Streamlit สำหรับสร้างเว็บแอปพลิเคชัน
import pandas as pd # นำเข้าไลบรารี pandas สำหรับจัดการข้อมูล
import joblib # นำเข้าไลบรารี joblib สำหรับโหลดโมเดล
import os # นำเข้าไลบรารี os สำหรับการทำงานกับระบบปฏิบัติการ (เช่น ตรวจสอบไฟล์)

# ตรวจสอบว่าโมเดลถูกบันทึกหรือไม่
model_path = 'model/sale_regression_model.pkl' # กำหนดพาธของไฟล์โมเดล
if not os.path.exists(model_path): # ตรวจสอบว่าไฟล์โมเดลมีอยู่หรือไม่
    st.error(f"Error: Model file '{model_path}' not found. Please ensure the model is saved correctly in Step 7.") # แสดงข้อความผิดพลาดถ้าไม่พบไฟล์โมเดล
else:
    # โหลดโมเดลที่บันทึกไว้
    loaded_model = joblib.load(model_path) # โหลดโมเดลจากไฟล์

    st.title('Sales Prediction Web App 📈') # ตั้งชื่อหน้าเว็บแอปพลิเคชัน
    st.write('กรุณากรอกงบประมาณการโฆษณาในแต่ละช่องทางเพื่อพยากรณ์ยอดขาย') # แสดงข้อความแนะนำผู้ใช้

    # สร้างช่องกรอกข้อมูลสำหรับผู้ใช้
    tv_budget = st.number_input('งบประมาณ TV (ล้านบาท)', min_value=0.0, max_value=300.0, value=150.0, step=1.0) # สร้างช่องกรอกงบประมาณ TV
    radio_budget = st.number_input('งบประมาณ Radio (ล้านบาท)', min_value=0.0, max_value=50.0, value=20.0, step=0.1) # สร้างช่องกรอกงบประมาณ Radio
    newspaper_budget = st.number_input('งบประมาณ Newspaper (ล้านบาท)', min_value=0.0, max_value=100.0, value=10.0, step=0.1) # สร้างช่องกรอกงบประมาณ Newspaper

    # ปุ่มสำหรับพยากรณ์
    if st.button('พยากรณ์ยอดขาย'): # ถ้าผู้ใช้คลิกปุ่ม 'พยากรณ์ยอดขาย'
        # สร้าง DataFrame จากข้อมูลที่ผู้ใช้กรอก
        unseen_data_for_prediction = pd.DataFrame({
            'TV': [tv_budget],
            'Radio': [radio_budget],
            'Newspaper': [newspaper_budget]
        }) # สร้าง DataFrame จากข้อมูลที่ผู้ใช้กรอก

        # ทำการพยากรณ์
        predicted_sales = loaded_model.predict(unseen_data_for_prediction)[0] # ใช้โมเดลทำนายยอดขาย

        st.success(f'ยอดขายที่คาดการณ์: {predicted_sales:,.2f} ล้านบาท') # แสดงผลลัพธ์ยอดขายที่คาดการณ์
if st.button("🏠 กลับหน้าหลัก"): # สร้างปุ่ม 'กลับหน้าหลัก'
    st.switch_page("app.py") # เปลี่ยนหน้าไปยัง 'app.py'

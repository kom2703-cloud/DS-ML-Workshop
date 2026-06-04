 # คำสั่ง magic command เพื่อเขียนโค้ดลงในไฟล์ classify_redbull_sale.py
import streamlit st # นำเข้าไลบรารี Streamlit สำหรับสร้าง Web Application
import pandas as pd # นำเข้าไลบรารี pandas สำหรับจัดการข้อมูล
import joblib # นำเข้าไลบรารี joblib สำหรับโหลดโมเดล

# --- 1. Load Model and Encoders ---
# Make sure these files are in the same directory as your streamlit app.py
try: # พยายามโหลดไฟล์โมเดลและ encoders
    loaded_model = joblib.load('model/redbull_best_classify_model.pkl') # โหลดโมเดล
    loaded_encoders = joblib.load('redbull_encoders.pkl') # โหลด encoders
    st.success('✅ โหลดโมเดลและ Encoder สำเร็จ!') # แสดงข้อความสำเร็จ
except FileNotFoundError: # หากไม่พบไฟล์
    st.error("⚠️ ไม่พบไฟล์โมเดลหรือ Encoder. โปรดตรวจสอบว่าไฟล์ 'redbull_best_classify_model.pkl' และ 'redbull_encoders.pkl' อยู่ในโฟลเดอร์เดียวกับไฟล์ app.py ของคุณ") # แสดงข้อความผิดพลาด
    st.stop() # หยุดการทำงานของ Streamlit

# Get feature names from the notebook context (hardcoded for the app's simplicity)
features = ['Region', 'Product_Variant', 'Channel', 'Unit_Price', 'Marketing_Spend'] # กำหนดชื่อ features ที่ใช้ในโมเดล

# --- 2. Streamlit App Layout ---
st.set_page_config(page_title="Red Bull High Sales Predictor", layout="centered") # ตั้งค่าหน้าเว็บ Streamlit
st.title('🚀 Red Bull High Sales Predictor') # กำหนดชื่อเรื่องของแอป
st.markdown("เครื่องมือนี้ช่วยคาดการณ์ว่า Product จะมียอดขายสูงหรือไม่ ขึ้นอยู่กับงบประมาณการตลาดและปัจจัยอื่นๆ") # เพิ่มข้อความอธิบาย
st.markdown("--- รันโมเดล 'Decision Tree' ---") # เพิ่มข้อความแสดงโมเดลที่ใช้

st.header('ข้อมูลสำหรับทำนาย') # กำหนดหัวข้อสำหรับส่วนข้อมูลอินพุต

# --- 3. User Input Widgets ---

# Use loaded_encoders.classes_ to populate dropdowns dynamically
# Region
region_options = loaded_encoders['Region'].classes_ # ดึงตัวเลือกภูมิภาคจาก encoder
selected_region = st.selectbox('ภูมิภาค (Region)', region_options) # สร้าง selectbox สำหรับภูมิภาค

# Product Variant
product_options = loaded_encoders['Product_Variant'].classes_ # ดึงตัวเลือกประเภทผลิตภัณฑ์จาก encoder
selected_product = st.selectbox('ประเภทผลิตภัณฑ์ (Product Variant)', product_options) # สร้าง selectbox สำหรับประเภทผลิตภัณฑ์

# Channel
channel_options = loaded_encoders['Channel'].classes_ # ดึงตัวเลือกช่องทางจาก encoder
selected_channel = st.selectbox('ช่องทางการตลาด (Channel)', channel_options) # สร้าง selectbox สำหรับช่องทางการตลาด

# Unit Price
unit_price = st.number_input('ราคาต่อหน่วย (Unit Price)', min_value=1.0, value=42.0, step=0.1) # สร้าง number_input สำหรับราคาต่อหน่วย

# Marketing Spend
marketing_spend = st.number_input('งบประมาณการตลาด (Marketing Spend)', min_value=1000.0, value=195000.0, step=1000.0) # สร้าง number_input สำหรับงบประมาณการตลาด


# --- 4. Prediction Button ---
if st.button('ทำนายโอกาสขายสูง'): # หากผู้ใช้คลิกปุ่ม 'ทำนายโอกาสขายสูง'
    # --- 5. Preprocess Input Data ---
    # Encode categorical features
    ch_enc = loaded_encoders['Channel'].transform([selected_channel])[0] # เข้ารหัสช่องทางที่เลือก
    pr_enc = loaded_encoders['Product_Variant'].transform([selected_product])[0] # เข้ารหัสประเภทผลิตภัณฑ์ที่เลือก
    re_enc = loaded_encoders['Region'].transform([selected_region])[0] # เข้ารหัสภูมิภาคที่เลือก

    # Create DataFrame for model input (ensure column order matches training data)
    model_input_df = pd.DataFrame([{
        'Region': re_enc,
        'Product_Variant': pr_enc,
        'Channel': ch_enc,
        'Unit_Price': unit_price,
        'Marketing_Spend': marketing_spend,
    }], columns=features) # สร้าง DataFrame สำหรับอินพุตโมเดล

    # --- 6. Make Prediction ---
    prob = loaded_model.predict_proba(model_input_df)[0][1]  # Probability of High_Sales (class 1) # ทำนายความน่าจะเป็นของยอดขายสูง
    predicted_class = loaded_model.predict(model_input_df)[0] # ทำนายคลาส

    # --- 7. Display Results and Recommendation ---
    st.subheader('ผลการทำนาย') # กำหนดหัวข้อสำหรับส่วนผลการทำนาย

    if predicted_class == 1: # หากทำนายว่าเป็นยอดขายสูง
        st.success(f"**ผลลัพธ์:** มีโอกาส 'ยอดขายสูง' ({prob:.2%}) 🎉") # แสดงข้อความยอดขายสูง
    else: # หากทำนายว่าเป็นยอดขายต่ำ
        st.info(f"**ผลลัพธ์:** มีโอกาส 'ยอดขายต่ำ' ({1-prob:.2%}) 📉") # แสดงข้อความยอดขายต่ำ

    st.write(f"ความน่าจะเป็นที่จะมียอดขายสูง: **{prob:.2%}**") # แสดงความน่าจะเป็นในรูปแบบเปอร์เซ็นต์

    # Recommendation logic (consistent with the notebook)
    rec = '✅ แนะนำ: ควรลงทุนใน Channel/Product นี้' if prob >= 0.25 else '⚠️ พิจารณา: อาจต้องพิจารณาปัจจัยอื่นๆ หรือช่องทาง/สินค้าอื่น' # กำหนดคำแนะนำ
    st.markdown(f"**คำแนะนำ:** {rec}") # แสดงคำแนะนำ

    st.markdown("--- ยอดขายสูงถูกกำหนดเป็น Units_Sold >= 75th percentile ของข้อมูล ---") # เพิ่มข้อความอธิบายการกำหนด High Sales

if st.button("🏠 กลับหน้าหลัก"): # สร้างปุ่ม 'กลับหน้าหลัก'
    st.switch_page("app.py") # เปลี่ยนหน้าไปยัง 'app.py'

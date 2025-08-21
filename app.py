# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Настройка страницы
st.set_page_config(page_title="DebtRecovery Predictor", layout="centered")
st.title("⚖️ DebtRecovery Predictor")
st.markdown("Оценка перспективности судебного взыскания долгов")

try:
    # Загрузка модели и данных
    with open('model_rf.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    
    df = pd.read_csv('synthetic_court_cases.csv')
    
    # Информация о модели
    with st.sidebar:
        st.header("📊 О модели")
        st.success("**Точность: 82.6%**")
        st.write("Обучена на 8000+ делах")
        st.write("**Топ-3 фактора:**")
        st.write("1. Тип должника (25.9%)")
        st.write("2. Практика судьи (19.4%)") 
        st.write("3. Наличие залога (11.4%)")
        st.write(f"Успешных дел: {df['result'].mean():.1%}")

    # Основной интерфейс
    st.header("🧮 Калькулятор вероятности")
    
    with st.form("case_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            court_region = st.selectbox("Регион суда", [
                'Москва', 'СПб', 'Новосибирская обл.', 'Свердловская обл.',
                'Респ. Татарстан', 'Краснодарский край', 'Нижегородская обл.'
            ], index=0)
            
            debtor_type = st.selectbox("Тип должника", [
                'ФЛ', 'ИП', 'ООО (микро)', 'ООО (малое)', 'ООО (среднее)'
            ], index=0)
            
            claim_amount = st.number_input("Сумма иска (руб.)", 
                min_value=10000, value=150000, step=10000)
        
        with col2:
            has_pledge = st.radio("Наличие залога", ["Нет", "Да"], index=0)
            has_guarantor = st.radio("Наличие поручителя", ["Нет", "Да"], index=0)
            judge_grant_rate = st.slider("Ставка судьи", 0.3, 0.9, 0.6, 0.05)
        
        submitted = st.form_submit_button("🎯 Рассчитать вероятность", use_container_width=True)

    # Обработка прогноза
    if submitted:
        # Подготовка данных
        input_data = pd.DataFrame([{
            'court_region': court_region,
            'debtor_type': debtor_type,
            'judge_id': 'judge_default',
            'judge_grant_rate': judge_grant_rate,
            'claim_amount': claim_amount,
            'has_pledge': 1 if has_pledge == "Да" else 0,
            'has_guarantor': 1 if has_guarantor == "Да" else 0
        }])
        
        try:
            # Прогноз
            X_cat = encoder.transform(input_data[['court_region', 'debtor_type', 'judge_id']])
            X_num = input_data[['judge_grant_rate', 'claim_amount', 'has_pledge', 'has_guarantor']]
            X_input = np.hstack([X_num, X_cat])
            
            probability = model.predict_proba(X_input)[0][1]
            
            # Показ результатов
            st.success("## 📊 Результаты прогноза")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Вероятность успеха", f"{probability:.1%}")
            with col2:
                result = "Удовлетворено" if probability > 0.5 else "Отказано"
                st.metric("Прогнозируемый исход", result)
            
            # Объяснение факторов
            st.info("### 📝 Факторы влияния:")
            
            if debtor_type == 'ФЛ':
                st.write("🔻 **Тип должника (ФЛ)**: снижает вероятность")
            else:
                st.write("🔺 **Тип должника (бизнес)**: повышает вероятность")
            
            if has_pledge == "Да":
                st.write("🔺 **Залог**: значительно увеличивает шансы")
            
            if has_guarantor == "Да":
                st.write("🔺 **Поручитель**: увеличивает вероятность")
                
        except Exception as e:
            st.error("Ошибка прогнозирования")
            st.write(str(e))

except Exception as e:
    st.error("❌ Ошибка загрузки приложения")
    st.info("""
    **Убедитесь что в папке есть:**
    - model_rf.pkl (модель)
    - encoder.pkl (энкодер)
    - synthetic_court_cases.csv (данные)
    """)
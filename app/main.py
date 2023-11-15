###

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import operator

model = pickle.load(open('/model/model.pkl', 'rb'))

scaler = pickle.load(open('/model/scaler.pkl', 'rb'))

###

def recode(Age,Size,Sex,PrimarySite,TcatG,Tcat,HistLaurenGroup,Grade):
    Sex_Female = True if Sex == '女' else False
    
    PrimarySite_L = True if PrimarySite == '下部（L）' else False
    PrimarySite_M = True if PrimarySite == '中部（M）' else False
    PrimarySite_U = True if PrimarySite == '上部（U）' else False
    PrimarySite_nan = True if PrimarySite == '其他' else False
    
    TcatG_T1 =  True if TcatG == 'T1' else False
    TcatG_T2 =  True if TcatG == 'T2' else False
    TcatG_T3 =  True if TcatG == 'T3' else False
    TcatG_T4 =  True if TcatG == 'T4' else False
    
    Tcat_T1a = True if Tcat == 'T1a' else False
    
    HistLaurenGroup_Intestinal = True if HistLaurenGroup == '肠型' else False
    
    Grade_G3 = True if Grade == 'G3（低/未分化）' else False
    
    X_test = [[Age,Size,Sex_Female,
             PrimarySite_L,PrimarySite_M,PrimarySite_U,PrimarySite_nan,
             TcatG_T1, TcatG_T2,TcatG_T3, TcatG_T4, Tcat_T1a, 
             HistLaurenGroup_Intestinal,Grade_G3]]
    
    return X_test

###

def predict(X_test):
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    
    st.write("淋巴结转移:red[**阳性**]可能性", model.predict_proba(X_test_scaled)[0][1])
    st.write("淋巴结转移:blue[**阴性**]可能性", model.predict_proba(X_test_scaled)[0][0])
  
    output = ':red[**阳性**]' if y_pred == 1 else ':blue[**阴性**]'
    return output

###
def main():
    
    st.title('亚洲人群胃癌淋巴结转移预测模型')

    st.write('基于机器学习方法，利用[SEER数据库](https://seer.cancer.gov/registries/data.html)中4658例亚洲人胃癌患者（1988-2020）构建的淋巴结转移预测模型。\
        本模型**仅用于科学研究与教学**，未经过前瞻性临床试验验证，未得到临床应用批准。')

    st.divider()

    Age = st.slider('**年龄（岁）**',
                    min_value = 19, 
                    max_value = 89)

    Sex = st.radio("**性别**",
                   ["男", "女"]
                   )

    Size = st.slider('**肿瘤大小（mm）**', 
                     min_value = 1,
                     max_value = 670)

    PrimarySite = st.radio("**肿瘤位置**",
                           ["上部（U）", "中部（M）", "下部（L）", "其他"],
                           captions = ["贲门/胃底", "胃体/大弯侧/小弯侧", "胃窦/幽门", "未知或涵盖多个范围"]
                           )

    TcatG = st.radio("**肿瘤浸润深度（AJCC 8e）**",
                     ["T1", "T2", "T3", "T4"],
                     captions = ["粘膜/粘膜下层", "固有肌层", "浆膜下层", "浆膜层或侵犯临近器官"]
                     )

    Tcat = st.radio("**早癌浸润深度**",
                    ["T1a", "T1b", "其他", "非T1"],
                    captions = ["粘膜层", "粘膜下层"]
                    )
    
    st.radio("**远处转移**",
             ["无"],
             captions = [":red[***本模型不适用于伴有远处转移的胃癌。***]"])
    
    HistLaurenGroup = st.radio("**Lauren分型**",
                               ["肠型", "弥漫型", "混合型", "其他", "未知"]
                               )

    Grade = st.radio("**病理分级**",
                     ["G1", "G2", "G3", "未知"],
                     captions = ["高分化", "中分化", "低/未分化"]
                     )

    st.divider()
    
    if "disabled" not in st.session_state:
        st.session_state['disabled'] = False
    
    st.checkbox('**我理解本模型仅用于科学研究与教学。**',
                key="disabled"
                )
    
    if st.button("**预测**",
                 disabled=operator.not_(st.session_state.disabled)
                 ):
        
        X_test = recode(Age,Size,Sex,PrimarySite,TcatG,Tcat,HistLaurenGroup,Grade)
        output = predict(X_test)
        st.success('预测结果：淋巴结转移为 {}'.format(output))
    
if __name__=='__main__':
    main()

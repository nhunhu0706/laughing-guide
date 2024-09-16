import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
import cv2
from PIL import Image
from deepface import DeepFace


df = pd.read_csv('py4ai-score.csv')
for i in range(1,11):
  df.loc[:,f'S{i}'] = df.loc[:,f'S{i}'].fillna(0)
df['BONUS'] = df['BONUS'].fillna(0)
df['REG-MC4AI'] = df['REG-MC4AI'].fillna('N')
def Classgroup(row):
  if 'CTIN' in row['CLASS']:
    return 'Chuyên Tin'
  if 'CTRN' in row['CLASS']:
    return 'Trung Nhật'
  if 'CT' in row['CLASS']:
    return 'Chuyên Toán'
  if 'CL' in row['CLASS']:
    return 'Chuyên Lý'
  if 'CA' in row['CLASS']:
    return 'Chuyên Anh'
  if 'CV' in row['CLASS']:
    return 'Chuyên Văn'
  if 'SN' in row['CLASS'] or 'TH' in row['CLASS']:
    return 'Tích Hợp/Song Ngữ'
  if 'CSD' in row['CLASS']:
    return 'Sử Địa'
  if 'CH' in row['CLASS']:
    return 'Chuyên Hóa'
  return 'Khác'

df['CLASS-GROUP']=df.apply(Classgroup,axis=1)

tab1, tab2, tab3, tab4, tab5 = st.tabs(['Danh sách','Biểu đồ','Phân Nhóm','Phân loại','Xem điểm'])
with tab1:
  col1, col2, col3, col4 = st.columns(4)
  with col1:
      st.write('Giới tính')
      m = st.checkbox('Nam',True)
      M = 'M' if m else None
      f = st.checkbox('Nữ',True)
      F = 'F' if f else None
      df1 = df[df['GENDER'].str.contains(f'{M}|{F}')]
  with col2:
      grade = st.radio('Khối lớp', ('Tất cả', 'Lớp 10', 'Lớp 11','Lớp 12'), horizontal=False)
      G = np.where(grade == 'Lớp 10','10',np.where(grade == 'Lớp 11','11',np.where(grade=='Lớp 12','12','1')))
      df1 = df1[df1['CLASS'].str.contains(f'{G}')]
  with col3:
      room = st.selectbox('Phòng',('A114','A115','Tất cả'))
      R = np.where(room == 'A114','114',np.where(room =='A115','115','11'))
      df1 = df1[df1['PYTHON-CLASS'].str.contains(f'{R}')]
  with col4:
      time = st.multiselect('Buổi',['Sáng','Chiều'])
      TIME = np.where(time == ['Sáng'],'S',np.where(time == ['Chiều'],'C','-'))
      df1 = df1[df1['PYTHON-CLASS'].str.contains(f'{TIME}')]
  st.write('Lớp chuyên')
  c1,c2,c3,c4,c5 = st.columns(5)
  with c1:
      v = st.checkbox('Văn',True)
      V = 'Văn' if v else None
      t = st.checkbox('Toán',True)
      T = 'Toán' if t else None
  with c2:
      l = st.checkbox('Lý',True)
      L = 'Lý' if l else None
      h = st.checkbox('Hóa',True)
      H = 'Hóa' if h else None
  with c3:
      a = st.checkbox('Anh',True)
      A = 'Anh' if a else None
      tin = st.checkbox('Tin',True)
      TIN = 'Tin' if tin else None
  with c4:
      sd = st.checkbox('Sử Địa',True)
      SD = 'Sử' if sd else None
      tn = st.checkbox('Trung Nhật',True)
      TN = 'Trung' if tn else None
  with c5:
      ts = st.checkbox('TH/SN',True)
      TS = 'Tích' if ts else None
      k = st.checkbox('Khác',True)
      K = 'Khác' if k else None
  df1 = df1[df1['CLASS-GROUP'].str.contains(f'{V}|{T}|{L}|{H}|{A}|{TIN}|{SD}|{TN}|{TS}|{K}')]
  st.write('Số HS',len(df1),f'({len(df1[df1["GENDER"].str.contains('M')])} nam, {len(df1[df1["GENDER"].str.contains('F')])} nữ)')
  ma = max(df1['GPA']) if not df.empty else np.nan
  mi = min(df1['GPA']) if not df.empty else np.nan
  me = round(sum(df1['GPA'])/len(df1['GPA']),2) if not df.empty else np.nan
  st.write('GPA:cao nhất',ma,'thấp nhất',mi,'trung bình',me)
  df1
with tab2:
  t1,t2 = st.tabs(['Số lượng HS','Điểm'])
  with t1:
    fig = px.pie(df, names='PYTHON-CLASS')
    st.plotly_chart(fig)
    st.success('Kết luận: Số học sinh ở 2 buổi gần bằng nhau, nên giờ học là hợp lý, đáp ứng được nhu cầu của tất cả học sinh')
    figgroup = px.pie(df, names='CLASS-GROUP')
    st.plotly_chart(figgroup)
    st.success("""
              Kết luận: 
              - Học sinh khối Chuyên Toán có xu hướng quan tâm lớp AI nhất
              - Học sinh khối Chuyên Trung Nhật có xu hướng không hứng thú với lớp AI nhất
              - Học sinh khối Chuyên Lý, Chuyên Anh có xu hướng hứng thú với lớp AI hơn học sinh khối Chuyên Tin
              """)
    figgender = px.pie(df,names='GENDER')
    st.plotly_chart(figgender)
    st.success('Kết luận: Nhìn chung học sinh nam quan tâm đến lớp AI hơn học sinh nữ')
  with t2:
    session = st.radio('Điểm từng session',['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','GPA'],horizontal=True)
    figpoint = px.box(df,x='PYTHON-CLASS',y=session,color='GENDER')
    st.plotly_chart(figpoint)
    st.info('Kết luận: Nhìn chung học sinh nam học tốt hơn học sinh nữ và lớp 114 đạt điểm cao hơn lớp 115')
    figgpa = px.box(df,x='CLASS-GROUP',y='GPA')
    st.plotly_chart(figgpa)
    st.info("""
               Kết luận: 
               - khối Chuyên Tin học tốt nhất
               - khối Chuyên Toán, Chuyên Lý vẫn có phần lớp học sinh đảm bảo đậu lớp AI
               - 3 khối Tích Hợp/Song Ngữ, Chuyên Sử Địa, Trung Nhật có phân bố điểm GPA thấp nhất - học môn PythonAI không hiệu quả
               - Chuyên Văn có tỉ lệ đậu 100%
               - Chuyên Anh có dải điểm phân bố rộng nhất và tương đối đều nhất
               """)
    df2 = df[['GPA','CLASS-GROUP','GENDER']]
    df2 = df2.groupby(['CLASS-GROUP','GENDER']).mean()
    df2.reset_index(inplace=True)
    figgroupgpa = px.bar(df2,x='CLASS-GROUP',y='GPA',color='GENDER',barmode='group')
    st.plotly_chart(figgroupgpa)
    y = [df[f'S{i}'].mean() for i in range(1,11)]
    figsession = px.bar(df2,x=['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10'],y=y,labels={'x':'index'})
    st.plotly_chart(figsession)
with tab3:
  df['S-AVG']=df.loc[:,['S1','S2','S3','S4','S5','S7','S8','S9']].mean(axis=1)
  co1, co2 = st.columns(2)
  with co1:
    k = st.slider('Số nhóm',2,5,3,step=1)
  with co2:
    option = st.multiselect('Chọn đặc trưng',['S6','S10','S-AVG'],['S6','S10','S-AVG'],key='unique_key_1')
    X = df.loc[:,option].to_numpy()
    dfX = df.loc[:,option]
    dfX['PASS']=np.where(df['GPA']>=6,'Đậu','Rớt')
  def kmeans(X, k):
    history = []
    icenters = np.random.choice(X.shape[0], k, replace=False)
    centers = X[icenters]
    while True:
      d = np.zeros((X.shape[0], k))
      for i in range(X.shape[0]):
        for j in range(k):
          d[i][j] = np.linalg.norm(X[i] - centers[j])
      y = np.argmin(d, axis=1)
      history.append((centers, y))
      centers_new = np.zeros((k, X.shape[1]))
      for j in range(k):
        centers_new[j] = np.mean(X[y == j], axis=0)
      if np.array_equal(centers, centers_new):
        return history
      centers = centers_new
  
  history = kmeans(X, k)
  centers, y = history[-1]
  dfX['LABELS']=y
  lb = {0:'Nhóm 1',
        1:'Nhóm 2',
        2:'Nhóm 3',
        3:'Nhóm 4',
        4:'Nhóm 5'}
  dfX['GROUP']=dfX['LABELS'].map(lb)
  dfX['NAME']=df['NAME']
  if len(option) == 2:
    fig1 = px.scatter(dfX,x=dfX.iloc[:,0],y=dfX.iloc[:,1],color='GROUP',hover_name='PASS',labels={'x':f'{option[0]}','y':f'{option[1]}'})
    fig1.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig1)
  elif len(option) == 3:
    fig1 = px.scatter_3d(dfX,x=dfX.iloc[:,0],y=dfX.iloc[:,1],z=dfX.iloc[:,2],color='GROUP',hover_name='PASS',labels={'x':f'{option[0]}','y':f'{option[1]}','z':f'{option[2]}'})
    fig1.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig1)
  
    for i in range(k):
      ix = np.where(y==i)
      st.markdown(f"""
                  **{lb[i]}**
                  - GPA cao nhất <span style='color:red'>{df.loc[ix,'GPA'].max()}</span>
                  - GPA thấp nhất <span style='color:green'>{df.loc[ix,'GPA'].min()}</span>
                  - GPA trung bình <span style='color:blue'>{round(df.loc[ix,'GPA'].mean(),2)}</span>
                  """, unsafe_allow_html=True)
      cl1, cl2 = st.columns(2)
      with cl1:
        dfX.loc[ix[0],option]
      with cl2:
        if len(option)==2:
          fig = px.scatter(dfX,x=dfX.iloc[ix[0],0],y=dfX.iloc[ix[0],1],hover_name=dfX.loc[ix,'NAME'],color=dfX.loc[ix,'GROUP'],labels={'x':f'{option[0]}','y':f'{option[1]}'})
          fig.add_trace(go.Scatter(x=[centers[i][0]],y=[centers[i][1]],mode='markers',marker=dict(symbol="diamond", size=5, color='cyan'),name='center'))
          st.plotly_chart(fig)
        elif len(option)==3:
          fdf = dfX.iloc[ix[0]]
          fig1 = px.scatter_3d(fdf,x=fdf.iloc[:,0],y=fdf.iloc[:,1],z=fdf.iloc[:,2],color='GROUP',hover_name='NAME',labels={'x':f'{option[0]}','y':f'{option[1]}','z':f'{option[2]}'})
          fig1.add_trace(go.Scatter3d(x=[centers[i][0]],y=[centers[i][1]],z=[centers[i][2]],mode='markers',marker=dict(symbol="diamond", size=5, color='cyan'),name='center'))
          st.plotly_chart(fig1)
with tab4:
  options = st.multiselect('Chọn đặc trưng',['S6','S10','S-AVG'],['S6','S10','S-AVG'],key='unique_key_2')
  x = df[options]
  Y = np.where(dfX['PASS'] == 'Đậu', 1, 0)
  if len(options)==2:
    model = LogisticRegression()
    model.fit(x, Y)
    w = model.coef_[0]
    b = model.intercept_[0]
    w1, w2 = w
    fig2 = px.scatter(dfX,x=dfX.iloc[:,0],y=dfX.iloc[:,1],color='PASS',labels={'x':f'{options[0]}','y':f'{options[1]}'})
    x1 = np.array([0,10])
    x2 = -(w1*x1 + b)/w2
    fig2.add_trace(go.Scatter(x=x1,y=x2,mode='lines', line=dict(color='red', width=2)))
    fig2.update_layout(showlegend=False)
    st.plotly_chart(fig2)
    acc = round(model.score(x,Y),2)
    st.write('Độ chính xác',acc)
  if len(options)==3:
    fig2 = px.scatter_3d(dfX,x=dfX.iloc[:,0],y=dfX.iloc[:,1],z=dfX.iloc[:,2],color='PASS',labels={'x':f'{options[0]}','y':f'{options[1]}','z':f'{options[2]}'})
    model = LogisticRegression()
    model.fit(x, Y)
    w = model.coef_[0]
    b = model.intercept_[0]
    w1,w2,w3 = w
    x = x.to_numpy()
    x1 = np.linspace(x[:, 0].min(), x[:, 0].max(), 100)
    x2 = np.linspace(x[:, 1].min(), x[:, 1].max(), 100)
    xx1,xx2 = np.meshgrid(x1,x2)
    x3 = - (w1* xx1 + w2 * xx2 + b) / w3
    fig2.add_trace(go.Surface(x=x1,y=x2,z=x3))
    fig2.update_layout(showlegend=False)
    st.plotly_chart(fig2)
    acc = round(model.score(x,Y),2)
    st.write('Độ chính xác',acc)
with tab5:
  nan = np.full(len(df), np.nan).tolist()
  nan[0] = 'regconition.jpg'
  df['image']=nan
  img_file_buffer = st.camera_input("Take a picture")
if img_file_buffer is not None:
    img = Image.open(img_file_buffer)
    img = np.array(img)
    for idx, row in df.iterrows():
      img_path = row['image']
      if pd.notna(img_path):
        verification = DeepFace.verify(img, img_path, enforce_detection=False)
        if verification["verified"]:
          st.success(f"""{row['NAME']}
                          điểm của bạn:
                    S1: {row['S1']}, 
                    S2: {row['S2']}, 
                    S3: {row['S3']}, 
                    S4: {row['S4']},
                    S5: {row['S5']},
                    S6: {row['S6']},
                    S7: {row['S7']},
                    S8: {row['S8']},
                    S9: {row['S9']},
                    S10: {row['S10']},
                    GPA: {row['GPA']}""" )


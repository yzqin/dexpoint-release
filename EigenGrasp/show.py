import pickle
with open('/home/wyk/下载/EigenGrasp/grasp_model.pkl', 'rb') as f:
    data=pickle.load(f)

print(data)
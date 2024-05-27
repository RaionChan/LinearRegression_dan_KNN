import pandas as pd

pizza = {'diameter': [6, 8, 10, 14, 18],
            'harga': [7, 9, 13, 17.5, 18]}

pizza_df = pd.DataFrame(pizza)
print(pizza_df)




import matplotlib.pyplot as plt

pizza_df.plot(kind='scatter', x='diameter', y='harga')

plt.title('Perbandingan Diameter dan Harga Pizza')
plt.xlabel('Diameter (inch)')
plt.ylabel('Harga (dollar)')
plt.xlim(0, 25)
plt.ylim(0, 25)
plt.grid(True)
plt.show()




import numpy as np

X = np.array(pizza_df['diameter'])
y = np.array(pizza_df['harga'])

print(f'X: {X}')
print(f'y: {y}')

X = X.reshape(-1, 1)
X.shape

X


Pizza


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)





X_vis = np.array([0, 25]).reshape(-1, 1)
y_vis = model.predict(X_vis)

plt.scatter(X, y)
plt.plot(X_vis, y_vis, '-r')



plt.xlabel('Diameter (inch)')
plt.ylabel('Harga (dollar)')
plt.xlim(0, 25)
plt.ylim(0, 25)


plt.show()








diameter_pizza = np.array([12, 20, 23]).reshape(-1, 1)
diameter_pizza

pizza
variance_x = np.var(X.flatten(), ddof=1)
print(f'variance: {variance_x}')
pizza

np.cov(X.flatten(), y)



covariance_xy = np.cov(X.flatten(), y)[0][1]
print(f'covariance: {covariance_xy}')





print(f'slope: {slope}')




from sklearn.metrics import r2_score

intercept = np.mean(y) - slope * np.mean(X)
print(f'intercept: {intercept}')



diameter_pizza = np.array([12, 20, 23]).reshape(-1, 1)
diameter_pizza



prediksi_harga



for dmtr, hrg in zip(diameter_pizza, prediksi_harga):
    print(f'Diameter: {dmtr} prediksi harga: {hrg}')


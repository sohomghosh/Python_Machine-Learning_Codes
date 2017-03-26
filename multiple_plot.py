import matplotlib.pyplot as plt

erp,=plt.plot(selected_data2_erp['WorkExp'],selected_data2_erp['Salary'],color='red')
fin,=plt.plot(selected_data2_fin['WorkExp'],selected_data2_fin['Salary'],color='blue')
bna,=plt.plot(selected_data2_bna['WorkExp'],selected_data2_bna['Salary'],color='green')

plt.legend((erp,fin,bna),('ERP','Financial Analyst','Business Analyst'),loc='lower left',fontsize=8)
plt.xlabel('Work Exp')
plt.ylabel('Salary')
plt.show(block=True)

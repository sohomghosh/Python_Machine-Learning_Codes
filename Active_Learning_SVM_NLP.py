from sklearn import svm
import numpy as np
import math
fp=open("keywords.txt",'r')
fe=[]
for line in fp:
	tk=line.split()
	fe.append(str(tk[0]).lower())
fp.close()
window_size=5
fp1=open("C:DataSource.xml",'r')
fp2=open("C:Train_features.csv",'w')
labl=[]
features_train = np.zeros(shape=(0,len(fe)))
for line in fp1:
	featur=[0]*len(fe)
	if "<location>" in line:
		tk_loca=line.split("<location>")
		tk_loca_ri=str(tk_loca[0]).split(" ")
		tk_loca_ri_ws=tk_loca_ri[len(tk_loca_ri)-window_size-1:len(tk_loca_ri)-1]
		tk_loca_en=line.split("</location>")
		tk_loca_lf=str(tk_loca_en[1]).split(" ")
		tk_loca_lf_ws=tk_loca_lf[0:window_size]
		tk_loca_uni=set(tk_loca_ri_ws).union(set(tk_loca_lf_ws))
		for word in tk_loca_uni:
			i=0
			for wo in fe:
				if str(word).lower() ==str(wo):
					featur[i]=1
				i=i+1
		fp2.write(str(featur)+",1\n")
		features_train = np.vstack([features_train, featur])
		labl.append(1) #Location treating as one
	if "<organization>" in line:
		tk_org=line.split("<organization>")
		tk_org_ri=str(tk_org[0]).split(" ")
		tk_org_ri_ws=tk_org_ri[len(tk_org_ri)-window_size-1:len(tk_org_ri)-1]
		tk_org_en=line.split("</organization>")
		tk_org_en=str(tk_org_en[1]).split(" ")
		tk_org_en_ws=tk_org_en[0:window_size]
		tk_org_uni=set(tk_org_ri_ws).union(set(tk_org_en_ws))
		for word in tk_org_uni:
			i=0
			for wo in fe:
				if str(word).lower() ==str(wo):
					featur[i]=1
				i=i+1
		fp2.write(str(featur)+",0\n")
		features_train = np.vstack([features_train, featur])
		labl.append(0) #Organization treating as zero
fp1.close()
fp2.close()

features_test = np.zeros(shape=(0,len(fe)))
fp3=open("C:\\Users\\SatyakiBh\\Desktop\\Upwork\\UD.xml",'r')
fp4=open("C:\\Users\\SatyakiBh\\Desktop\\Upwork\\Test_features.csv",'w')
test_sents=[]
for line in fp3:
	featur=[0]*len(fe)
	if "<Toponyme>" in line:
		test_sents.append(str(line))
		tk_loca=line.split("<Toponyme>")
		tk_loca_ri=str(tk_loca[0]).split(" ")
		tk_loca_ri_ws=tk_loca_ri[len(tk_loca_ri)-window_size-1:len(tk_loca_ri)-1]
		tk_loca_en=line.split("</Toponyme>")
		tk_loca_lf=str(tk_loca_en[1]).split(" ")
		tk_loca_lf_ws=tk_loca_lf[0:window_size]
		tk_loca_uni=set(tk_loca_ri_ws).union(set(tk_loca_lf_ws))
		for word in tk_loca_uni:
			i=0
			for wo in fe:
				if str(word).lower() ==str(wo):
					featur[i]=1
				i=i+1
		fp4.write(str(featur)+",???\n")
		features_test = np.vstack([features_test, featur])
fp3.close()
fp4.close()


clf = svm.SVC(probability=True)
#k=5 taken in densty.py may be changed if required
#From density.py code of the paper "Active Learning for Entity Filtering in Microblog Streams"
def jaccquard_similarity(a, b):
    if len(b) == 0 or len(a) == 0: return 0.0
    return len(set(a).intersection(b))*1./len(set(a).union(set(b)))


def similarityMatrix(features):
	a = np.zeros((len(features), len(features)), dtype=np.float)
	for i in range(0,len(features),1):
		for j in range(0,len(features),1):
			a[i,j]=jaccquard_similarity(features[i],features[j])
	return a
def density(matrix, row):
   return np.mean(matrix[row,:])

def k_density(matrix, row, k=5):
   r = matrix[row,:]
   return np.mean(np.sort(r[1:k+1])[::-1])

def margin_density(distance, matrix, row):##distance is distance of method-1,row is row number,matrix is similarity matrix
   return (1-density(matrix, row)*(1-distance))

def margin_k_density(distance, matrix, row, k=5):#distance is distance of method-1,row is row number,matrix is similarity matrix
   return (1-k_density(matrix, row, k)*(1-distance))



clf.fit(features_train, labl)

probabs=clf.predict_proba(features_test)

#Method-1 Uncertainity based Margin Sampling #Call method_one_ms(probabs)
def method_one_ms(probab):
	uncertainity_test=[]
	for [a,b] in probab:
		uncertainity_test.append(1-math.fabs(a-b))
	return uncertainity_test

#Method-2 Density
def method_two_density(probab,features_tes):#Call method_two_density(probabs,features_test)
	density_test=[]
	row=0
	mat=similarityMatrix(features_tes)
	#print mat
	for ft in features_tes:
		uncertainity_test=method_one_ms(probab)
		dist=uncertainity_test[row]
		margin_k_dens=margin_k_density(dist,mat,row,k=5)
		row=row+1
		density_test.append(margin_k_dens)
	return density_test


#Method-3 Ranking Margin Sampling (method-1) based on density (method-2)
def method_three_rerank(probab,features_tes):#Call method_three_rerank(probabs,features_test)
	uncertainity_test=method_one_ms(probab)
	density_test=method_two_density(probab,features_tes)
	ranked_ms=[x for (y,x) in sorted(zip(density_test,uncertainity_test))]
	return ranked_ms


###Model updation
def method1(features_train,labl,features_test):
	ACTIVE = True
	clf.fit(features_train, labl)
	probabs=clf.predict_proba(features_test)
	while ACTIVE:
		ms=method_one_ms(probabs)
		a=sorted(ms,reverse=True)[0:5]#Number of questions may be changed by changing this number 5
		inx=[]
		for i in a:
			inx.append(ms.index(i))
		#Ask 5 queries, append to features_train run clf.fit, print probabilities new, make active=false
		print "Asking Questions for Method-1:"
		labels=[]
		#modi_feat=np.zeros(shape=(0,len(features_test[1])))
		modi_feat=features_train
		#new_feat= np.zeros(shape=(0,len(features_test[1])))
		for i in list(set(inx)):
			input_ans = input("Please enter response (0-organization or 1-location) for \n"+str(test_sents[i])+"\n")
			ft=np.asarray(features_test[i])
			ft=np.reshape(ft,(1,217))
			#np.concatenate((features_train, ft),axis=0)
			modi_feat=np.concatenate((modi_feat, ft),axis=0)
			labels.append(input_ans)
		clf.fit(modi_feat, labl+labels)
		probabs=clf.predict_proba(features_test)
		print probabs	
		choice=input("Would you like to repeat 0-Yes 1-No\n")
		if choice==1:
			ACTIVE=False
			break

def method2(features_train,labl,features_test):
	ACTIVE=True
	labels=labl
	clf.fit(features_train, labels)
	probabs=clf.predict_proba(features_test)
	while ACTIVE:
		den=method_two_density(probabs,features_test)
		#Ask 5 queries, append to features_train run clf.fit, print probabilities new, make active=false
		a=sorted(den,reverse=True)[0:5]#Number of questions may be changed by changing this number 5
		inx=[]
		for i in a:
			inx.append(den.index(i))
		#Ask 5 queries, append to features_train run clf.fit, print probabilities new, make active=false
		print "Asking Questions for Method-2:"
		for i in list(set(inx)):
			input_ans = input("Please enter response (0-organization or 1-location) for \n"+str(test_sents[i])+"\n")
			features_train=np.vstack([features_train, features_test[i]])
			labels.append(input_ans)
		clf.fit(features_train, labels)
		probabs=clf.predict_proba(features_test)
		print probabs	
		choice=input("Would you like to repeat 0-Yes 1-No\n")
		if choice==1:
			ACTIVE=False
			break

def method3(features_train,labl,features_test):
	ACTIVE=True
	labels=labl
	clf.fit(features_train, labl)
	probabs=clf.predict_proba(features_test)
	while ACTIVE:
		rerank=method_three_rerank(probabs,features_test)
		#Ask 5 queries, append to features_train run clf.fit, print probabilities new, make active=false
		a=sorted(rerank,reverse=True)[0:5]#Number of questions may be changed by changing this number 5
		inx=[]
		for i in a:
			inx.append(rerank.index(i))
		#Ask 5 queries, append to features_train run clf.fit, print probabilities new, make active=false
		print "Asking Questions for Method-3:"
		for i in list(set(inx)):
			input_ans = input("Please enter response (0-organization or 1-location) for \n"+str(test_sents[i])+"\n")
			features_train=np.vstack([features_train, features_test[i]])
			labels.append(input_ans)
		clf.fit(features_train, labels)
		probabs=clf.predict_proba(features_test)
		print probabs
		choice=input("Would you like to repeat 0-Yes 1-No\n")
		if choice==1:
			ACTIVE=False
			break

mtd_ch=input("Enter Choice 1 for Margin Sampling, 2 for Margin Density Sampling, 3 for Reranking \n")
if mtd_ch==1:
	method1(features_train,labl,features_test)
if mtd_ch==2:
	method2(features_train,labl,features_test)
if mtd_ch==3:
	method3(features_train,labl,features_test)

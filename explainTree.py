from sklearn import tree
import graphviz
import pickle

chosen_drug = "Cannabis"

with open('./data/respondents_train.pickle', 'rb') as f:
    respondents_train = pickle.load(f)

with open('./data/respondents_test.pickle', 'rb') as f:
    respondents_test = pickle.load(f)

with open('./data/drugs_train.pickle', 'rb') as f:
    drugs_train = pickle.load(f)

with open('./data/drugs_test.pickle', 'rb') as f:
    drugs_test = pickle.load(f)


# model = tree.DecisionTreeClassifier()
# model = model.fit(respondents_train, drugs_train[chosen_drug].astype('int'))
# with open('tree.pkl','wb') as f:
#     pickle.dump(model,f)
with open('tree.pkl', 'rb') as f:
    model = pickle.load(f)


###### Create tree representation

def displayTree():
    print(list(respondents_train.columns))
    text_representation = tree.export_text(model, feature_names=list(respondents_train.columns))
    print(text_representation)

    dot_data = tree.export_graphviz(model, out_file=None,
                       feature_names=respondents_train.columns,  
                       class_names=['0','1'],
                       filled=True)

    graph = graphviz.Source(dot_data, format="png") 
    graph.view()


    ##### Show samples


    # print(respondents_test.iloc[8:9])
    # print(drugs_test[chosen_drug].iloc[8:9])

    # drug_pred = model.predict(respondents_test.iloc[8:9])

    # print(drug_pred)



def CalculateGiniTopNode():
    column_label = []
    for i in range(len(respondents_train['Country'])):
        column_label.append((respondents_train['Country'].iloc[i], drugs_train[chosen_drug].iloc[i]))

    #threshold = (0.24923 + 0.21128)/2
    threshold = (0.96082 + 0.24923)/2
    print(len(respondents_train['Country']))

    class1=0
    class0=0
    count=0
    for i in column_label:
        
        count+=1
        
        if(i[1]==1):
            class1+=1
        else:
            class0+=1
    print(1-(class1/count)**2 -(class0/count)**2)

def CalculateValuesLeftMostNode(threshold_escore):
    results0 = []
    results1 = []
    for i in range(len(respondents_train['Country'])):
        if(respondents_train.iloc[i]['Country']<=0.605):
            if(respondents_train.iloc[i]['ID']<=421):
                if(respondents_train.iloc[i]['Cscore']<=-0.274): 
                    if(respondents_train.iloc[i]['Escore']<=threshold_escore): 
                        results0.append((respondents_train.iloc[i], drugs_train[chosen_drug].iloc[i]))
                    else:
                        results1.append((respondents_train.iloc[i], drugs_train[chosen_drug].iloc[i]))
    print("class 1")
    print("number of elements", len(results1))
    for i in results1:
        print()
        print(i)

    print("\n\nclass 0")
    print("number of elements", len(results0))
    for i in results0:
        print()
        print(i)



# import numpy as np
# with open("Country_column.txt", "w") as f:
#     #print(str(np.array(respondents_test['Country'])))
#     f.write(str((sorted(column_label, key=lambda item: item[0], reverse=True))))

displayTree()
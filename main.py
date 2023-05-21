import pandas as pd
from ModelClass import Model

# Import csv file
data = pd.read_csv('data.csv')

# Initialize model class
model = Model(data)

while True:
    print('''
1. CAR
2. ROA
3. NPF
4. NPF_Net
5. FDR
6. BOPO
7. NOM
8. APYD_Terhadap_Aktiva_Produktif
9. Short_Term_Mistmach
    ''')

    try:
        model.showGraph(int(input('Choose a number from above option: ')))
    except:
        print('That\'s not a number')
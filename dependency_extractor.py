import xml.etree.ElementTree as ET
import csv
import pandas as pd
import numpy as np

# Load the .xmi file
tree = ET.parse('petclinic2_java.xmi')
root = tree.getroot()

# Define the namespace
namespace = {'xsi':'http://www.w3.org/2001/XMLSchema-instance'}

# Define the desired elements and their corresponding CSV headers
elements = [
    {'name': 'bodyDeclarations', 'xpath': './/bodyDeclarations[@xsi:type="java:MethodDeclaration"]', 'csv_header': 'Method Name'},
    {'name': 'bodyDeclarations', 'xpath': './/bodyDeclarations[@xsi:type="java:ConstructorDeclaration"]', 'csv_header': 'Method call'},
]

# Open the CSV file for writing
with open('raw_output.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['Method','Class', 'Usages','Parameter'])
    writer.writeheader()

    # Iterate over the desired elements and extract the data
    for element in elements:
        for item in root.findall(element['xpath'], namespace):
            if 'originalCompilationUnit' in str(item.attrib):
                parameters = item.findall(".//parameters", namespaces=namespace)
                parameter_count = len(parameters)
                data = {
                    'Method': item.attrib['name'].strip(),
                    'Class': item.attrib['originalCompilationUnit'].strip(),
                    'Usages' : item.attrib['usages'].strip() if 'usages' in str(item.attrib) else '',
                    'Parameter' : parameter_count
                }
            writer.writerow(data)
    
df = pd.read_csv('raw_output.csv')
df.Usages = df.Usages.apply(str)
for item in root.findall('.//compilationUnits',namespace):
    if 'types' in str(item.attrib):
       #print(item.attrib['name'],item.attrib['types'])
       df.Usages = df.Usages.apply(lambda x: x.replace(item.attrib['types'] ,item.attrib['name']) if item.attrib['types'] in x else x)

df.Usages = df.Usages.apply(lambda x: ','.join([ y.split('.java',1)[0] for y in x.split(' ')]))
#print( [(item.attrib['name'],item.attrib['originalCompilationUnit'] if 'originalCompilationUnit' in str(item.attrib) else '') for item in root.findall('.//ownedElements[@xsi:type="java:ClassDeclaration"]',namespace)])

for item in root.findall('.//ownedElements[@xsi:type="java:ClassDeclaration"]',namespace):
    if 'originalCompilationUnit' in str(item.attrib):
        #print(item.attrib['name'],item.attrib['originalCompilationUnit'].strip())
        df=df.replace(item.attrib['originalCompilationUnit'].strip(),item.attrib['name'])
for item in root.findall('.//ownedElements[@xsi:type="java:InterfaceDeclaration"]',namespace):
    if 'originalCompilationUnit' in str(item.attrib):
        #print(item.attrib['name'],item.attrib['originalCompilationUnit'].strip())
        df=df.replace(item.attrib['originalCompilationUnit'].strip(),item.attrib['name'])
df=df.drop_duplicates()
#df.to_csv('output2.csv')

new_rows = []
for index, row in df.iterrows():
    usages = row['Usages']
    if isinstance(usages, str):
        usages_list = usages.split(',')
        for usage in usages_list:
            new_row = row.copy()
            new_row['Usages'] = usage.strip()
            new_rows.append(new_row)

# Create a new DataFrame from the replicated rows
new_df = pd.DataFrame(new_rows)

#new_df.to_csv('output3.csv')

df2 = pd.DataFrame(new_df.pivot_table(index = new_df.columns.tolist(), aggfunc ='size'))


df2.to_csv('raw_output2.csv')

df2 = pd.read_csv('raw_output2.csv')

df2 = df2.rename(columns={"0":"CallCount"})
df2 = df2.replace(np.nan,"NA")
df2["CallCount"] = np.where(df2["Usages"]=="NA",0,df2["CallCount"])

# Save dependency data 
df2.to_csv('dependency_data.csv',index=False)







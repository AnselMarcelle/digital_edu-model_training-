import pandas as pd
df = pd.read_csv('train.csv')
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
#print(df['career_start'].describe())

'''
#Гипотеза 2: Люди, у которых большое количество подписчиков, с большей вероятностью купят курс.
#Преобразуем количество подписчиков в бинарную переменную (1 - у пользователя много подписчиков, 0 - мало)
def followers_cleaner(count):
    if count > 1000:
        return 1
    return 0

df['followers_count'] = df['followers_count'].apply(followers_cleaner)
print(df.groupby(by='followers_count')['result'].mean())
'''

'''
#Гипотеза 2: Люди с высоким образованием с большей вероятностью купят курс.
# Группируем данные по уровню образования и выводим среднее значение переменной result для каждой группы
print(df.groupby(by='education_status')['result'].mean())
'''

'''
#Гипотеза 3: Люди, которые уже имеют работу, могут быть менее склонны к покупке курса.
# Преобразуем данные о наличии работы в бинарную переменную (1 - у пользователя есть работа, 0 - нет)
def occupation_cleaner(occupation):
    if occupation != 'NaN':
        return 1
    return 0

df['occupation_type'] = df['occupation_type'].apply(occupation_cleaner)
print(df.groupby(by='occupation_type')['result'].mean())
'''

'''
#Гипотеза 4: Люди, живущие в крупных городах, с большей вероятностью купят курс.
# Преобразуем данные о городе в бинарную переменную (1 - пользователь живет в крупном городе, 0 - в маленьком)
def city_cleaner(city):
    if city in ['Moscow', 'Saint Petersburg', 'Kazan', 'Novosibirsk', 'Yekaterinburg']:
        return 1
    return 0

df['city'] = df['city'].apply(city_cleaner)
print(df.groupby(by='city')['result'].mean())
'''

'''
#Гипотеза 5: Люди, чей язык обучения отличается от языка своей страны, менее склонны к покупке курсов
def lang_diff(row):
    langs = row['langs'].split(',')
    if len(langs) == 1:
        return 0
        native_lang = row['city'].split(',')[1]
        if native_lang in langs:
            return 0
    return 1

df['lang_diff'] = df.apply(lang_diff, axis=1)
print(df.groupby(by='lang_diff')['result'].mean())
'''



#Обучение модели
df['langs'] = df['langs'].apply(lambda lang: lang.count(';')+1)
df.drop(['id','sex','bdate','has_photo','has_mobile','followers_count','graduation','education_form','relation','education_status','life_main','people_main','city','last_seen','occupation_type','occupation_name','career_start','career_end'], axis=1, inplace=True)
x = df.drop('result', axis=1)
y = df['result']
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.25)

sc = StandardScaler()
train_x = sc.fit_transform(train_x)
test_x = sc.transform(test_x)

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(train_x, train_y)

y_pred = classifier.predict(test_x)
accuracy = accuracy_score(test_y, y_pred)
print("Точность: ", accuracy)

# Предлагаю соблюдать следующие правила написания текста статьи, чтобы все было однотипно и понятно:
```
1. Прямо как в Python делаем отступы с помощью Tab в каждом новом логическом разделе.
```
```
2. Во всех формулах отделяем математические операции с помощью пробелов. Или, проще говоря, следуем PEP 8: https://peps.python.org/pep-0008/
```
```
3. Имена ссылкам даем говорящие, чтобы легко в них ориентироваться, например, \label{Main Hamiltonian}.
```
```
4. Ссылки на литературу потом оформим в отдельном файле, но название ссылок тоже даем говорящие. Я предлагаю так: \cite{Creutz1980} - фамилия первого автора и год.
```
```
5. Все графики подписываем говорящими названиями и храним в папке Figures, тоже самое с исходниками к ним.
```
```
6. Тексты статей храним в папке References, bib файл тоже там поместим.
Соблюдение этих правил несколько лет назад при написании статей мне бы сейчас настолько жизнь упростило... Давайте сразу учтем горький опыт.
```


# PIMC {Seva, could you additionally write the full title ?:)}

PIMC-py is the python code folder. It contains model classes, Monte-Carlo simulation code and the first draft of the neural network code. Written by Seva. {Seva, could you write in more detail?}. The last update was 2023-09-23, edited by Dima {I just made the folder:}). 

PIMC-cpp is the cpp code folder. It contains model classes. File info.txt contains 1) compile and launch commands, 2) file names and a brief description of their content, 3) files structure. Header files for convenience have an id at the beginning (for example, 0-0, 1-0, etc.). The gsl lib is used. Test cpp files contain the code that allows you to check the corresponding header files. File main.cpp contains only the general structure of the included libs and the header files. The last update was 2023-09-27, edited by Dima.


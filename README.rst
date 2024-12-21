|test| |codecov| |docs|

.. |test| image:: https://github.com/intsystems/ProjectTemplate/workflows/test/badge.svg
    :target: https://github.com/intsystems/ProjectTemplate/tree/master
    :alt: Test status
    
.. |codecov| image:: https://img.shields.io/codecov/c/github/intsystems/ProjectTemplate/master
    :target: https://app.codecov.io/gh/intsystems/ProjectTemplate
    :alt: Test coverage
    
.. |docs| image:: https://github.com/intsystems/ProjectTemplate/workflows/docs/badge.svg
    :target: https://intsystems.github.io/ProjectTemplate/
    :alt: Docs status


.. class:: center

    :Название исследуемой задачи: Классификация траекторий динамических систем с помощью физически-информированных нейросетей
    :Тип научной работы: НИР
    :Автор: Терентьев Александр Андреевич
    :Научный руководитель: д. ф.-м. н. Стрижов Вадим Викторович

Abstract
========

Целью работы является проверка гипотезы о возможности классификации тракеторий физических систем по лагранжианам, а также задачей сатвить предложить метод для данной классификации. Идея состоит в том, чтобы сначала восстановить лагранжиан системы, которая могла бы породить данную траеторию. Лагранжиан восстанавливается с помощью физически-информированных Лагражевых нейронных сетей. Предложена норма, которая вводится на пространстве лагранжианов. Она используется как метрика для метрических методов классификации. В основном для многомерных классификаций используется светрочные нейронные сети, в предположении, что существуют статистические свзяи, между ближайшими точками временного ряда. Данные методы хорошо себя показывают во многих задачах, где эти связи являются прелбладающими. Но для физических систем такие методы не подходят, и предлагается использовать знания о физических связах систем, для выбранных Ларанжевых нейронных сетей этим знанием является закон сохранения энергии.

Code link
========
1) https://github.com/intsystems/Terentev-BS-Thesis/tree/master/code/main.ipynb

Installation
========

1. `git clone` this repository.
2. Create new `conda` environment and activate it
3. Run 




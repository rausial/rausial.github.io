# Reseña Diseño de tableros de información (mostrar datos para entender de un vistazo)

1. TOC
{:toc}

## Introducción

Los tableros de información son recursos de visualización de información que nos permiten darle seguimiento a procesos que necesitamos mantener en buen curso, sí como un coche o un avión donde mientras manejamos tenemos que estar al tanto de varias cosas como la gasolina, nuestra velocidad, y si todo parece estar funcionando adecuadamente para llegar a nuestro destino sin problemas. Si vemos, por ejemplo, que se está acabando la gasolina entonces analizamos si es necesario ir a cargar en ese momento o podemos ir más adelante. En todo caso el medidor de gasolina nos da información sobre nuestro auto que nosotros evaluamos y decidimos si hay que actuar. Si el tablero prende el foco de que se está acabando la gasolina, entonces tenemos que re-evaluar y quizá lo más recomendable, ahora sí, sea ir a una gasolinera. La CONABIO tiene una gran acerbo de información sobre la biodiversidad en México, que abarca multiples temas desde muchas perspectivas, y en la actualidad la biodiversidad del planeta es un tema de preocupación mundial. Por esta razón la CONABIO está renovando la forma de transmitir mucha de la información que alberga para dar acceso más eficaz a la gente en general y a los tomadores de decisiones que inciden en la conservación de nuestro medio ambiente en particular. Dentro del abanico de posibilidades para transmitir esta información están las herramientas de visualización de información. Este texto es el inicio de una serie de textos que documentarán el proceso para diseñar estas herramientas. Comienzo revisando el libro [Information Dashboard Design](http://stephen-few.com/idd.php) de [Stephen Few](http://stephen-few.com/).

El libro presenta el proceso para diseñar un tablero desde el mero principio, esto es, desde ubicar para qué quiero el tablero, donde tenemos que preguntarnos cosas básicas que muchas veces perdemos de vista en la prisa por empezar. En los primeros dos capítulos define brevemente lo que son los tableros y qué no hacer cuando diseñamos uno. Los siguientes cuatro capítulos no son exclusivamente de tableros, hay varios elementos que son útiles para otro tipo de proyectos, en particular es una invitación constante a que nos detengamos a pensar el '¿para qué queremos esto?'. En este texto nos concentraremos en estos capítulos, que según lo veo son los fundamentos para tener un proyecto. Todo el trabajo que se debe hacer en esta etapa es de pizarrón, ni siquiera involucra diseño de visualizaciones, es sobre las definiciones a las que tenemos que llegar para poder empezar a imaginar cómo se ve lo que necesitamos.

Tema | Punto de vista | Perspectiva del futuro | Acciones
-----|----------------|------------------------|---------
Biodiversidad | Conservación | Tendencias | Aumentar presupuesto de áreas naturales protegidas, apoyar agricultura sustentable, etc.

En particular con productos de visualización, es fácil perdernos en lo flashy, o sea, en creer que lo principal sea conseguir quién nos haga un diseño deslumbrante. Lo deslumbrante, sin embargo, debe ser un efecto colateral derivado del concepto y la efectividad para transmitir la información que queremos transmitir para que las personas que utilizan el tablero hagan su trabajo mejor informados, más alertas sobre las cosas que hay que atender. El libro es un poco manual y un poco una revisión de la teoría que sustenta el manual. Mantiene un mensaje central, que tiene muchas versiones en distintas disciplinas -Occam's razor en estadística, Less is More en arquitectura, KISS (Keep It Simple Stupid) en desarrollo de software- que en este libro se expresa más rimbombante como: 'Elocuencia mediante la simplicidad' (Capítulo 6: Achieving Eloquence through simplicity).

Las secciones que revisamos en este texto elaboran sobre:

* ¿Qué y para qué es un tablero de información?
* Lo que no debemos hacer cuando diseñemos un tablero
* ¿Quiénes son las personas que usaran la herramienta? y ¿por qué? o ¿para qué?
* ¿Cuáles son las fuentes de información disponibles?

Un tablero de información es, en principio, un artefacto para seguir algún proceso que nos interesa desde una perspectiva particular. Es un punto de acceso que condensa todo en una sola pantalla (no se vale que uno tenga que darle scroll). Es, en cierto sentido, un gran ejercicio de sintetización y de reflexión sobre qué es lo importante con respecto a lo que nos interesa. 

Los capítulos 1 y 2 de 'Information Dashboard Design' nos dan una idea general de la utilidad de los tableros de información, y nos alertan que hay muchas formas en las que puede ir mal un diseño. Esto última pasa porque quienes diseñaron el tablero no consideran cosas fundamentales. Con la clásica frase  'no todo lo que brilla es oro' señala que muchos diseños fallan por concentrarse en el factor '¡guau!' en vez de utilizar los elementos visuales basándose en principios de percepción visual.

Es importante entender para qué queremos un tablero de información, la respuesta de Stephen Few es: para mantenernos al tanto de una situación. Y, agrega, estar realmente al tanto de una situación funciona en tres niveles:

1. Percepción de los elementos
2. Comprensión
3. Visión del futuro

En el capítulo (?) Stephen Few aclara que si la información que se muestra en el tablero se actualiza con una frecuencia mayor a **un día** entonces no le llamará tablero. Pero esa restricción es sólo porque si algo se actualiza con menor frecuencia entonces el requisito de que todo se pueda ver y entender de un vistazo se relaja. 

El capítulo 2 se trata de una serie de ejemplos donde se pueden apreciar errores comunes que van desde llevar la metáfora del tablero de coche al extremo (ridículo) de querer que un tablero **de información** se vea como un tablero de coche. Hay otros ejemplos que muestran malas elecciones de gráficas, y como el afán de algún programador por hacer lucir el tablero por su diseño acaba siendo un desastre porque ni es diseñador y sólo mete ruido visual que no aporta nada de información. Es un capítulo para agarrar el *feeling* de lo que hay que evitar.

## Fundamentos para armar un buen tablero (capítulos 3 al 6)

Del capítulo 3 al 6 vamos, progresivamente, discutiendo cómo conceptualizar el producto que vamos a desarrollar. Empezando por lo muy elemental en términos del proyecto y terminando con los conceptos básicos de visualización que nos ayudarán a tomar decisiones sobre qué gráficas y cómo diseñarlas. En términos muy generales el flujo de los primeros capítulos es:

### Entender lo que se necesita (Capítulo 3)

Este capítulo inicia la discusión sobre cómo ir de lo general a lo particular. Las recomendaciones son útiles más allá de los tableros de información. Plantea algo muy obvio que, sin embargo, nos saltamos muchas veces. Definir ¿de qué va el proyecto? Y la respuesta debe definir a quiénes va dirigido, y cuáles son las necesidades de estas personas. Si no entendemos al público y lo que necesita entonces ¿cómo vamos a tener ideas claras sobre el diseño?

Para entender esto Stephen Few propone las siguientes preguntas:

* ¿Quiénes lo van a usar? Por ejemplo: Ciudadanía
* ¿Qué tan frecuentemente se va a actualizar? Por ejemplo: Diario
* ¿Qué va a monitorear y qué objetivos debe apoyar?
* ¿Qué preguntas debe responder el tablero? ¿Qué acciones se deben tomar de acuerdo a las respuestas?
* ¿Qué elementos de información debemos desplegar en el tablero? ¿Qué nos dice cada uno de esos elementos y por qué es importante? ¿A qué nivel se debe resumir o detallar  la información para dar el panorama necesario?
* ¿Cuáles de estos elementos de información son los más importantes para lograr nuestros objetivos?
* ¿Cuáles son los agrupamientos lógicos que se podrían utilizar para organizar los elementos de información? ¿A qué grupo pertenece cada elemento de información?
* ¿Cuáles son las comparaciones más útiles que nos permitirán ver los elementos de información en el contexto más significativo?


### Consideraciones fundamentales (Capítulo 4)

Definir el espectro del público y el medio de presentación del tablero, contexto de las métricas

#### Contexto de las métricas

Dar contexto es, por ejemplo, comparar la medición actual contra algo, puede ser la misma medida en el pasado o con respecto a un objetivo o para ver si es anómala. 

Este capítulo habla sobre cómo implementar el mensaje. Si nuestro objetivo es, por ejemplo, ¿Cómo está progresando México hacia el cumplimiento de los compromisos 2030? Ahí ya tenemos una serie de metas contra las cuáles tendrían que ir comparadas las métricas. Pero incluso en este objetivo cabe detenerse a revisar el enfoque. Porque podríamos preguntarnos si lo queremos ver en términos absolutos, o términos de si México está mejorando (acelerando) su esfuerzo por cumplir con los objetivos.  

### El poder de la percepción visual (Capítulo 5)

Este es un capítulo que brevemente revisa la teoría sobre cómo funciona nuestro sistema de percepción visual y porque es tan poderoso en términos de adquisición de información.

# Reseña Diseño de tableros de información (mostrar datos para entender de un vistazo)

1. TOC
{:toc}

## Introducción

---

Los tableros de información son recursos de visualización de información que nos permiten darle seguimiento a procesos que necesitamos mantener en buen curso. Sí, como un coche o un avión donde mientras manejamos tenemos que estar al tanto de: la gasolina, nuestra velocidad, y, en general, si todo parece estar funcionando para llegar a nuestro destino sin problemas. Si vemos, por ejemplo, que se está acabando la gasolina, entonces analizamos si es necesario ir a cargar en ese momento o podemos ir más adelante. El medidor de gasolina nos da información sobre nuestro auto que evaluamos y decidimos si hay que actuar. Si el tablero prende el foco de que se está acabando la gasolina, entonces tenemos que re-evaluar y quizá lo más recomendable, ahora sí, sea ir a una gasolinera. 

---

La CONABIO tiene una gran acerbo de información sobre la biodiversidad en México, que abarca múltiples temas, desde muchas perspectivas. La biodiversidad del planeta es un tema de preocupación mundial. En este contexto la CONABIO está renovando la forma de transmitir mucha de la información que alberga para dar acceso más eficaz a la gente en general y a los tomadores de decisiones que inciden en la conservación de nuestro medio ambiente en particular. Dentro del abanico de posibilidades para transmitir esta información están las herramientas de visualización de información. Este texto es el inicio de una serie de textos para documentar el proceso de diseño de algunas de estas herramientas. Como punto de partida estamos revisando el libro [Information Dashboard Design](http://stephen-few.com/idd.php) de [Stephen Few](http://stephen-few.com/).

---

El libro presenta un camino para diseñar un tablero desde el mero principio. Esto es, comienza desde el trabajo que necesitamos realizar para definir bien para qué queremos el tablero. En los primeros dos capítulos define brevemente lo que son los tableros y qué no hacer cuando diseñamos uno. Los siguientes cuatro capítulos son particular es una invitación constante a que nos detengamos a pensar el '¿para qué queremos esto?'. En este texto nos concentraremos en estos capítulos, que plantean los fundamentos para tener un proyecto bien especificado. Todo el trabajo que se debe hacer en esta etapa es de pizarrón, ni siquiera involucra diseño de visualizaciones, es sobre las definiciones a las que tenemos que llegar para empezar a imaginar cómo se ve lo que necesitamos. El último de esta sección da una serie de lineamientos para empezar, ahora sí, con el diseño. 

---

#### Definición de tableros

El libro tiene una definición muy especifica de lo que es un tablero, que en nuestro caso CONABIO no se ajusta perfectamente, ya que Stephen Few requiere que el objetivo sea dar seguimiento a un proceso que requiera acción casi inmediata, donde al menos se requiere actualizar diario la información. El monitoreo de incendios caería en esa categoría, sin embrago, en temas de seguimiento de la conservación de la biodiversidad las temporalidades son distintas, ya que no tiene información que se actualice tan rápido y las acciones que se requieren son de mediano a largo plazo. Por otro lado, en el caso de nuestro proyecto es muy importante la perspectiva del mensaje, que creo se podría dividir en los elementos: tema, punto de vista, perspectivas del futuro, y acciones posibles. Por ejemplo: 

---

Tema | Punto de vista | Perspectiva del futuro | Acciones
-----|----------------|------------------------|---------
Biodiversidad | Conservación | Cobertura actual forestal y tendencias en la deforestación, pérdida de hábitat de especies protegidas | Aumentar presupuesto de áreas naturales protegidas, apoyar agricultura sustentable, etc.

---

Tener claros los objetivos nos protege de caer en un error común, en particular con productos de visualización, que es perdernos en lo flashy, o sea, buscar un diseño deslumbrante. Lo deslumbrante debe ser un efecto colateral derivado del concepto y la efectividad para transmitir la información que queremos transmitir para que las personas que utilizan el tablero hagan su trabajo mejor informados, más alertas sobre las cosas que hay que atender. 

El libro es un poco manual y un poco una revisión de la teoría que sustenta el manual. Mantiene un mensaje central, que tiene muchas versiones en distintas disciplinas -Occam's razor, Less is More, KISS (Keep It Simple Stupid)- que en este libro se expresa un poco rimbombantemente como: 'Elocuencia mediante la simplicidad' (Capítulo 6: Achieving Eloquence through simplicity).

Las secciones que revisamos en este texto elaboran sobre:

* ¿Qué y para qué es un tablero de información?
* Lo que no debemos hacer cuando diseñamos un tablero
* ¿Quiénes son las personas que usaran la herramienta? y ¿por qué? o ¿para qué?
* ¿Cuáles son las fuentes de información disponibles?

Un tablero de información es, en principio, un artefacto para seguir algún proceso que nos interesa desde una perspectiva particular. Es un punto de acceso que condensa todo en una sola pantalla (no se vale que uno tenga que darle scroll). Es, en cierto sentido, un ejercicio de sintetización y de reflexión sobre qué es lo importante para mantenernos al tanto de una situación. 

Los capítulos 1 y 2 de 'Information Dashboard Design' nos dan una idea general de la utilidad de los tableros de información, y nos alertan que hay muchas formas en las que puede ir mal un diseño. Esto último comúnmente pasa porque quienes diseñaron el tablero no consideran cosas fundamentales. Con la clásica frase  'no todo lo que brilla es oro' señala que muchos diseños fallan por concentrarse en el factor '¡guau!' en vez de utilizar los elementos visuales basándose en principios de percepción visual que se alineen con los objetivos de la herramienta.

Es importante entender cuándo queremos un tablero de información, la respuesta del libro es: cuando necesitamos mantenernos al tanto de una situación (situation awareness). Y, agrega, estar realmente al tanto de una situación funciona en tres niveles:

    1. Percepción de los elementos del ambiente
    2. Comprensión del la situación actual, y
    3. Proyección del estatus a futuro

El capítulo 2 se trata de una serie de ejemplos donde se pueden apreciar errores comunes que van desde llevar la metáfora del tablero de coche al extremo (ridículo) de querer que un tablero **de información** se vea como un tablero de coche. Hay otros ejemplos que muestran malas elecciones de gráficas, y como el afán de algún programador por hacer lucir el tablero por su diseño acaba siendo un desastre porque ni es diseñador y sólo mete ruido visual que no aporta nada de información. Es un capítulo para agarrar el *feeling* de lo que hay que evitar.

## Fundamentos para armar un buen tablero (capítulos 3 al 6)

La idea es seguir los capítulos del 3 al 6 para definir el proyecto que vamos a desarrollar. Empezando por lo fundamental y terminando con los conceptos básicos de visualización que nos ayudarán a tomar decisiones sobre qué gráficas y cómo diseñarlas. En breve el flujo de los primeros capítulos es:

1. ¿Qué deberíamos considerar en términos de objetivos   
2. 
### Entender lo que se necesita (Capítulo 3)

Este capítulo inicia la discusión sobre cómo ir de lo general a lo particular. Las recomendaciones son útiles más allá de los tableros de información. Plantea algo muy obvio que, sin embargo, nos saltamos muchas veces. Definir ¿de qué va el proyecto? Y la respuesta debe definir a quiénes va dirigido, y cuáles son las necesidades de estas personas. Si no entendemos al público y lo que necesita entonces ¿cómo vamos a tener ideas claras sobre el diseño?

El consejo general para empezar es: 'enfocarnos en los objetivos, no en el medio'. El tablero (o cualquier visualización para el caso) es un medio, es el mensajero, de ... ¿cuál es **el mensaje**?

Para entender a fondo nuestro mensaje el capítulo propone las siguientes preguntas:

1. ¿Qué va a monitorear y qué objetivos debe apoyar?
2. ¿Quiénes lo van a usar?
3. ¿Qué preguntas debe responder el tablero? 
4. ¿Qué acciones se deben tomar de acuerdo a las respuestas?
5. ¿Qué elementos de información debemos desplegar en el tablero? 
6. ¿Qué nos dice cada uno de esos elementos y por qué es importante? 
7. ¿A qué nivel se debe resumir o detallar la información para dar el panorama necesario?
8. ¿Cuáles de estos elementos de información son los más importantes para lograr nuestros objetivos?
9. ¿Cuáles son los agrupamientos lógicos que se podrían utilizar para organizar los elementos de información? 
10. ¿A qué grupo pertenece cada elemento de información?
11. ¿Cuáles son las comparaciones más útiles que nos permitirán ver los elementos de información en el contexto más significativo?
12. ¿Qué tan frecuentemente se va a actualizar?

Las preguntas del 1 al 4 son las que definirían en buena medida los objetivos fundamentales del tablero. Las siguientes nos ayudan a definir nuestro modelo de información.

### Consideraciones fundamentales (Capítulo 4)

Con una idea más clara de los objetivos y del tipo de información que vamos a manejar, la siguiente fase nos ayuda a estructurar mejor la propuesta. Y para sirve revisar las siguientes seis características, que ponen en contexto los objetivos generales para especificar los requerimientos del diseño. 

Característica | Descripción
---------------|------------
Tamaño de audiencia | Una persona<br>Múltiples personas con mismos requerimientos<br>Múltiples personas que requieren monitorear subconjuntos de datos distintos<br> 
Nivel de experiencia de usuarios | Novato<br>Conocedor<br>Experto
Frecuencia de actualización | Diario<br>Cada hora<br>Tiempo real
Plataforma tecnológica | Computadora de escritorio<br>Navegador web<br>Dispositivo móvil
Tipo de pantalla | Extra grande<br>Estándar<br>Pequeña<br>Variadas
Tipo de datos | Cuantitativos<br>Cualitativos

En este capítulo Stephen Few aclara que si la información que se muestra en el tablero se actualiza con una frecuencia mayor a **un día** entonces no le llamará tablero. Esa restricción es sólo porque si algo se actualiza con menor frecuencia entonces el requisito de que todo se pueda ver y entender de un vistazo se relaja. 

#### Contexto de las métricas

Dar contexto es comparar el estado actual contra algo, puede ser la misma medida en el pasado o con respecto a un objetivo o para ver si es anómala. 

El contexto que demos es una implementación del mensaje. Si nuestro objetivo es, por ejemplo, ¿cómo está progresando México hacia el cumplimiento de los compromisos de desarrollo sustentable para 2030? Ahí ya tenemos una serie de metas contra las cuales tendrían que ir comparadas las métricas. A partir de este objetivo podríamos preguntarnos, por ejemplo, si lo queremos ver en términos absolutos, o en términos de si México está mejorando (acelerando) su esfuerzo por cumplir con los objetivos. Estas comparaciones nos pueden hablar de si vamos bien o mal en algo, por eso es importante pensar contra qué debemos comparar. 

### Entrándole al poder de la percepción visual (Capítulo 5)

Este es un capítulo que brevemente revisa la teoría sobre cómo funciona nuestro sistema de percepción visual y porque es tan poderoso en términos de adquisición de información. En él podemos revisar rápidamente consideraciones sobre color, áreas, longitudes, gradientes, y qué tan efectivo es cada uno para transmitir información cuantitativa, y para comparar mediciones.

### Logrando elocuencia mediante la simplicidad (Capítulo 6)

En este ya entramos en el diseño propiamente del tablero. El reto es:

>Condensar una gran cantidad de datos en un espacio pequeño, y al mismo tiempo mantener claridad

y

>Poner todo lo necesario sin sacrificar significado

Previo a esto está otro reto básico (que no es exclusivo de los tableros):

>Elegir los datos adecuados para el objetivo


Los tableros son para decirle a la gente lo que está pasando, deben cumplir con su tarea de tal forma resaltan cualquier cosa que requiere la atención inmediata del usuario. Y dado tienen muy poco espacio para hacer esto requieren mucha precisión. 

Un tablero bien hecho debe comunicar información que está:

1. Excepcionalmente bien organizada
2. Condensada, principalmente en resúmenes
3. Específica de la tarea en curso y acomodada para comunicarle claramente a quienes lo usen
4. Desplegados usando, usualmente, medios conscisos y pequeños que comunican la información de la forma más clara y directa posible

Son ante todo una herramienta que da un vistazo de nivel alto a la información sobre el estado de las cosas. Deben proveer también acceso rápido y fácil a la información adicional que se necesita para responder a cualquier evento.

Mantener óptima la razón de datos a pixeles, buscando eliminar lo más posible los pixeles no-dato.

#### El diseño de un tablero es un proceso iterativo. 

* Creamos un ejemplo falso de nuestro tablero y lo mejoramos a través de una serie de revisiones, cada una seguida de evaluación nueva que nos lleva a un rediseño. 


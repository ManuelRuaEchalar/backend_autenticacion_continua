# Avance de tesis y solicitud de decisión

**Juan Ruiz Aechalar — 25/08/2026**

Detalle completo en `informe_director_2026-08-25.md` y documentos de respaldo.

---

## 1. Lo aprobado

**PI:** cómo influye la configuración de sensores (acelerómetro, giroscopio,
gestos táctiles) en el equilibrio entre efectividad de autenticación y consumo
de batería y RAM, en autenticación continua con aprendizaje federado sobre
móviles de gama media.

**H:** a más sensores, mayor efectividad y mayor consumo; mínimo 1 punto
porcentual de EER entre configuraciones extremas.

**Muestra:** 15-20 participantes, cada uno con su móvil, uso natural, 8-12
semanas. **Observaciones del comité:** replantear la hipótesis (general y ya
probada) y explorar otros sensores.

El artefacto se construyó completo y reportaba **EER 7.98%**. Pero el pool de
impostores venía de *otros* teléfonos. Si el modelo reconocía el **aparato** y
no la **persona**, ese número no significaba nada frente al ataque real: alguien
que coge tu propio móvil. Se recogieron impostores en el mismo dispositivo (16
y 17 de agosto, 4 voluntarios) para comprobarlo.

## 2. Lo que se midió

| prueba | resultado |
|---|---|
| control negativo (dueño contra sí mismo) | AUC 0.42-0.52 ✔ instrumentación sana |
| **dos personas, mismo móvil, tarea dirigida** | **AUC 0.972**; 0.997 en la condición dura |
| impostores del pool (otros aparatos) | FAR 1.4% / 0.0% |
| **impostores no vistos, mismo móvil** | **FAR 73.3% / 50.2%** |
| ...frente a la aceptación del **dueño** | 71.8% / 46.0% |
| varianza: persona / dispositivo / **resto** | 5.5% / 9.2% / **85.4%** |
| pool de 1→32 impostores, capacidad relevante | plana (0.59 → 0.52) |
| conjunto abierto (2 arquitecturas, HMOG) | EER 22% y 32% |
| ...las mismas en nuestros móviles | 1 de 4 medidas supera el azar |
| 66 ventanas de sensado equivalen a | **2.4 observaciones independientes** |
| **nueve formulaciones distintas** | todas en la misma banda |

Tres conclusiones:

1. **En el móvil 1 un impostor desconocido entraba más a menudo que el
   propietario.** La cabeza personal memorizó sus 584 ventanas de entrenamiento
   y puntuaba 0.999 todo lo demás: es un clasificador de conjunto cerrado
   aplicado a un problema de conjunto abierto.
2. **El 85.4% de la varianza no es persona ni aparato: es sesión** (postura,
   agarre, tarea, momento). De ahí sale mecánicamente por qué acumular más
   sensado no ayudaba.
3. **Que un discriminante lineal iguale a dos arquitecturas del estado del arte
   indica techo de datos, no de modelo.** Lo que domina no es el modelo: es el
   protocolo de recolección.

## 3. Por qué esto rompe el perfil aprobado

**La hipótesis deja de ser contrastable — saldría «no significativa» en los dos
escenarios posibles, por razones opuestas.** Con el protocolo estándar el EER
está saturado por memorización (AUC 0.99 ya con dos sensores): no hay recorrido
donde pueda aparecer el punto porcentual que la hipótesis exige. Con el
protocolo honesto, todo está cerca del azar.

**El eje de efectividad se queda sin variable dependiente.** Un cociente
coste/efectividad con numerador de ruido no es interpretable.

**Un participante por dispositivo confunde persona y aparato**, y el aparato
pesa más (9.2% frente a 5.5%).

**Y el sesgo de dispositivo crece con la configuración.** El magnetómetro lleva
calibración propia de cada unidad y mide un campo propio de cada sitio — las dos
cosas valen cero en el ataque real. Evaluado con impostores de otros teléfonos,
*acc+giro+mag* ganaría efectividad **por el atajo**, y se concluiría que
«compensa su coste de batería» por la razón equivocada.

## 4. Primer replanteamiento (18/08)

Manteniendo problema, objetivo y diseño aprobados, cambiando sólo la
operacionalización de la efectividad:

- **H1** — el coste marginal no depende del número de canales sino del
  **régimen de muestreo**: los de sondeo continuo imponen coste fijo; los gestos
  táctiles, dirigidos por eventos que el sistema ya genera, cuestan casi cero.
- **H2** — el equilibrio **depende del protocolo de evaluación**. Es una
  hipótesis de interacción: se contrasta como el término configuración ×
  protocolo del ANOVA de medidas repetidas que el perfil ya contempla.
- **H3** — un discriminante lineal extrae más señal que el modelo desplegado
  (0.785 frente a 0.332): separa «cuánta señal hay» de «cuánta extrae esta
  arquitectura».

Y el **magnetómetro** como cuarta configuración, respondiendo a la segunda
observación del comité: es de sondeo continuo (discrimina H1) y el más
específico de unidad y lugar (donde H2 predice la mayor brecha).

## 5. La objeción que cambió el planteamiento

> *Si el modelo no reconoce al usuario legítimo frente a un impostor no visto,
> ¿no es como inventar un vehículo volador que no vuela y medir cuántos litros
> consume?*

Es correcta para una parte. Quedan prohibidas: «la configuración B ofrece el
mejor equilibrio», «añadir el giroscopio mejora la autenticación» sin decir bajo
qué protocolo, y cualquier FAR presentado como el del despliegue.

Donde se rompe: **el aparato existe y funciona; falla una de sus salidas.** La
aplicación recolecta, entrena en el dispositivo, federa y evalúa — 30 rondas,
dos clientes, sin fallos. Medir su coste es legítimo; decir que autentica, no. Y
**el coste establece la envolvente dentro de la cual cualquier solución futura
tiene que caber**: no es cuánta gasolina gasta el coche que no vuela, es cuánto
empuje hace falta como mínimo.

**Consecuencia de estructura:** el trabajo debe **liderar con el resultado
negativo y su mecanismo**. Si lidera con las mediciones de recursos y menciona
de pasada que no autentica, el tribunal atacará con razón.

## 6. Segundo replanteamiento: entorno controlado

En lugar de 15-20 personas con su móvil en uso libre: **20-30 personas sobre dos
dispositivos comunes**, sesiones dirigidas con un minijuego de tecleo, repetidas
en días distintos.

No es una apuesta: es el protocolo del **único resultado limpio** de todo el
trabajo (AUC 0.972, mismo teléfono, misma tarea). Convierte en diseño lo que
allí ocurrió por casualidad.

| lo que rompió el estudio | con el diseño controlado |
|---|---|
| confound de dispositivo | persona × dispositivo **cruzado**, efectos separables |
| 85.4% de varianza de sesión | **colapsa**: misma tarea, misma postura |
| pool de 1-2 impostores | 20-30 identidades, partición disjunta por persona |
| prestar el móvil a desconocidos | innecesario |

Es además el protocolo de **HMOG y BehavePassDB**, las dos referencias del área.
**Lo que cuesta:** deja de ser uso ambiental (mide un límite superior), la
efectividad no generaliza a otros aparatos, y el federado sería simulado por
partición de identidad con validación de coste en hardware real.

## 7. Las rutas, y qué aporta cada una

| | corpus | efectividad | recursos | tipo de tesis | campo |
|---|---|---|---|---|---|
| **A** perfil original | su móvil, uso libre | ruido (medido) | real | **negativa** | alto |
| **B** post-objeción | el ya recogido | negativa + mecanismo | real | **negativa** | ninguno |
| **C** controlado | 20-30 pers., 2 móviles | plausible | real | positiva | ~75 h |
| **D** sólo HMOG | 100 sujetos, público | comparable | **ninguno** | aporte débil | ninguno |

**Qué le aporta C al aprendizaje federado** (la objeción evidente: dos móviles
son un corpus centralizado). La heterogeneidad no-IID —problema central de FL—
mezcla **persona**, **dispositivo** y **contexto**, y nadie las separa. Este
trabajo ya midió el reparto: **85% es contexto, no estructura de cliente.** El
diseño controlado fija la tarea (el 85.4% colapsa) y cruza persona con
dispositivo: queda un banco de pruebas con la no-IID **descompuesta**, que
ningún corpus público permite. E importa porque cada método de personalización
ataca una fuente distinta: la normalización local ataca el dispositivo, la
cabeza personal (ya implementada) ataca la persona — **y la cabeza personal
cuesta batería**. Nadie puede decidir hoy cuál sobra.

**Variante que recomiendo:** HMOG para efectividad y preentrenamiento (100
sujetos, comparable con la literatura) **+ corpus propio pequeño y cruzado**
(8-12 personas, 3-4 sesiones) sólo para recursos y descomposición. Baja el campo
de ~75 h a ~15-20 h sin perder potencia —el dispositivo es factor intra-sujeto—
y el FAR se reporta sobre 100 sujetos en vez de 25.

## 8. Cuatro decisiones que solicito

1. **¿Acepta el programa una tesis de resultado negativo?** Decide todo lo
   demás. Mejor saberlo antes de comprometer 60 días de campo.
2. **Cambio de naturaleza de la muestra** (§3.3, Anexo 3): de 15-20 dispositivos
   a N personas sobre 2 dispositivos comunes.
3. **El uso deja de ser natural.** Es la desviación más visible y la más
   defendible: es el protocolo de HMOG y BehavePassDB.
4. **¿Reportar efectividad sobre corpus público y recursos sobre hardware
   propio?** Abarata y refuerza, pero es un cambio de alcance.

---

*Estado del desarrollo:* módulo de medición de recursos reconstruido y
verificado en hardware —el anterior medía la batería por delta de porcentaje y
**669 de 676 medidas valían exactamente 0.0**—; modelo de datos del estudio
controlado probado sobre los 141 MB de campo reales sin pérdida de una fila; 109
pruebas unitarias y 17 en dispositivo, en verde.

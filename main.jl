using FileIO
using Images

# Cargar una imagen
madera = load("bloques/madera/madera.jpg");
piedra = load("bloques/piedra/piedra.jpg");

# Mostrar esa imagen, de cualquiera de estas dos formas
display(imagen)
# using ImageView; imshow(imagen);

# Que tipo tiene la imagen (es un array)
typeof(imagen)
# Array{RGB{Normed{UInt8,8}},2}:
#   Array bidimensional: Array{    ,2}
#   donde cada elemento es de tipo RGB{Normed{UInt8,8}}
# Tamaño de la imagen: el tamaño del array
size(imagen)
# Por ejemplo, para ver el primer pixel:
dump(imagen[1,1])
# Tipo: RGB{Normed{UInt8,8}}
typeof(imagen[1,1])
# Cada pixel tiene 3 campos: r,g,b
# Cada campo es del tipo indicado
#  Normed{UInt8,8}): 8 bits, normalizado entre 0 y 1
#  Por ejemplo, para comar la componente roja, de cualquiera de estas formas
imagen[1,1].r
red(imagen[1,1])
# Las otras dos componentes, de igual manera:
imagen[1,1].g
green(imagen[1,1])
imagen[1,1].b
blue(imagen[1,1])
# Para crear un elemento de tipo RGB, simplemente instanciar RGB indicando los 3 componentes, por ejemplo, el blanco:
RGB(1,1,1)


# Para extraer un canal de la imagen, hacer un broadcast de la operacion correspondiente que se realiza a un pixel, pero a toda la matriz
#  red(pixeles) -> devuelve la componente roja de ese pixel
#  red.(array de pixeles) -> devuelve un array del mismo tamaño, con las componente rojas de esos pixeles
# Por ejemplo:
matrizRojos = red.(imagen);
# Para construir una imagen solamente con ese canal, hacer una operacion de broadcast
#  RGB(         1,        0, 0 ) -> devuelve el color rojo (solo un pixel)
#  RGB.( [0.1, 0.5, 0.9], 0, 0 ) -> devuelve un array de 3 elementos (es decir, una imagen de 1 fila y 3 columnas) con esos colores. Esta linea es equivalente a:
#  RGB.( [0.1, 0.5, 0.9], [0, 0, 0], [0, 0, 0] )
# Por tanto, para construir la imagen solo con el canal rojo
imagenRojo = RGB.(matrizRojos,0,0)
# De esta forma, la imagen original se pueden extraer sus 3 canales (rojo, verde y azul) y recomponerla de la siguiente manera:
RGB.(red.(imagen), green.(imagen), blue.(imagen))

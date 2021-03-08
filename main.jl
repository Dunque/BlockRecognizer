using FileIO
using Images
using Statistics

# Functions that allow the conversion from images to Float64 arrays
imageToGrayArray(image:: Array{RGB{Normed{UInt8,8}},2}) = convert(Array{Float64,2}, gray.(Gray.(image)));
imageToGrayArray(image::Array{RGBA{Normed{UInt8,8}},2}) = imageToGrayArray(RGB.(image));
function imageToColorArray(image::Array{RGB{Normed{UInt8,8}},2})
    matrix = Array{Float64, 3}(undef, size(image,1), size(image,2), 3)
    matrix[:,:,1] = convert(Array{Float64,2}, red.(image));
    matrix[:,:,2] = convert(Array{Float64,2}, green.(image));
    matrix[:,:,3] = convert(Array{Float64,2}, blue.(image));
    return matrix;
end;
imageToColorArray(image::Array{RGBA{Normed{UInt8,8}},2}) = imageToColorArray(RGB.(image));

# Some functions to display an image stored as Float64 matrix
# Overload the existing display function, either for graysacale or color images
import Base.display
display(image::Array{Float64,2}) = display(Gray.(image));
display(image::Array{Float64,3}) = (@assert(size(image,3)==3); display(RGB.(image[:,:,1],image[:,:,2],image[:,:,3])); )

# Cargar una imagen
function loadFolderImages(folderName::String)
    isImageExtension(fileName::String) = any(uppercase(fileName[end-3:end]) .== [".JPG", ".PNG"]);
    images = [];
    for fileName in readdir(folderName)
        if isImageExtension(fileName)
            image = load(string(folderName, "/", fileName));
            # Check that they are color images
            @assert(isa(image, Array{RGBA{Normed{UInt8,8}},2}) || isa(image, Array{RGB{Normed{UInt8,8}},2}))
            # Add the image to the vector of images
            push!(images, image);
        end;
    end;
    # Convert the images to arrays by broadcasting the conversion functions, and return the resulting vectors
    return (images);
end;

function loadRedChannel(imageArray)
    result = [];
    for image in imageArray
        matrizR = red.(image);
        push!(result, matrizR);
    end;
    return result;
end;

function loadBlueChannel(imageArray)
    result = [];
    for image in imageArray
        matrizR = blue.(image);
        push!(result, matrizR);
    end;
    return result;
end;

function loadGreenChannel(imageArray)
    result = [];
    for image in imageArray
        matrizR = green.(image);
        push!(result, matrizR);
    end;
    return result;
end;

function displayImages(imageArrayR, imageArrayG, imageArrayB)
    for image in imageArrayR
        display(RGB.(image,0,0));
    end;
    for image in imageArrayG
        display(RGB.(0,image,0));
    end;
    for image in imageArrayB
        display(RGB.(0,0,image));
    end;
end;

madera = loadFolderImages("bloques/madera");
# piedra = load("bloques/piedra/piedra.jpg");

channelR = loadRedChannel(madera);
channelG = loadGreenChannel(madera);
channelB = loadBlueChannel(madera);

meanR = mean(mean.(channelR))
meanG = mean(mean.(channelG))
meanB = mean(mean.(channelB))

stdevR = mean(std.(channelR))
stdevG = mean(std.(channelG))
stdevB = mean(std.(channelB))
#displayImages(channelR, channelG, channelB);
# matrizG = green.(madera);
# matrizB = blue.(madera);
# # Para construir una imagen solamente con ese canal, hacer una operacion de broadcast
# #  RGB(         1,        0, 0 ) -> devuelve el color rojo (solo un pixel)
# #  RGB.( [0.1, 0.5, 0.9], 0, 0 ) -> devuelve un array de 3 elementos (es decir, una imagen de 1 fila y 3 columnas) con esos colores. Esta linea es equivalente a:
# #  RGB.( [0.1, 0.5, 0.9], [0, 0, 0], [0, 0, 0] )
# # Por tanto, para construir la imagen solo con el canal rojo
# imagenRojo = RGB.(matrizR,0,0)
# imagenVerde = RGB.(0,matrizG,0)
# imagenAzul = RGB.(0,0,matrizB)
#
# display(imagenRojo);
# display(imagenVerde);
# display(imagenAzul);
# # De esta forma, la imagen original se pueden extraer sus 3 canales (rojo, verde y azul) y recomponerla de la siguiente manera:
# RGB.(red.(madera), green.(madera), blue.(madera))

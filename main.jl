using FileIO
using Images
using Statistics
using Flux
using Random
using Plots
using BSON
using Zygote

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

function loadTrainingDataset()
    elementos=["madera","piedra"]
    targets=Array{Any,2}(undef, 0, 7);
    for cada_uno_de_los_elementos in elementos
        folder=string("bloques/",cada_uno_de_los_elementos)
        elemento = loadFolderImages(folder);
        channelR = loadRedChannel(elemento);
        channelG = loadGreenChannel(elemento);
        channelB = loadBlueChannel(elemento);
        for input = 1:length(channelR)
            line=[mean(channelR[input]) #=
            =# mean(channelG[input]) #=
            =# mean(channelB[input]) #=
            =# std(channelR[input]) #=
            =# std(channelG[input]) #=
            =# std(channelB[input]) #=
            =# cada_uno_de_los_elementos]
            targets=[targets;line]
        end;
    end;
    return targets
end;

function oneHotEncoding(feature::Array{Any,1}, classes::Array{Any,1})
    # Primero se comprueba que todos los elementos del vector esten en el vector de clases (linea adaptada del final de la practica 4)
    @assert(all([in(value, classes) for value in feature]));
    numClasses = length(classes);
    @assert(numClasses>1)
    if (numClasses==2)
        # Si solo hay dos clases, se devuelve una matriz con una columna
        oneHot = Array{Bool,2}(undef, size(feature,1), 1);
        oneHot[:,1] .= (feature.==classes[1]);
    else
        # Si hay mas de dos clases se devuelve una matriz con una columna por clase
        oneHot = Array{Bool,2}(undef, size(feature,1), numClasses);
        for numClass = 1:numClasses
            oneHot[:,numClass] .= (feature.==classes[numClass]);
        end;
    end;
    return oneHot;
end;
oneHotEncoding(feature::Array{Any,1}) = oneHotEncoding(feature::Array{Any,1}, unique(feature));

accuracy(outputs::Array{Bool,1}, targets::Array{Bool,1}) = mean(outputs.==targets);
function accuracy(outputs::Array{Bool,2}, targets::Array{Bool,2}; dataInRows::Bool=true)
    @assert(all(size(outputs).==size(targets)));
    if (dataInRows)
        # Cada patron esta en cada fila
        if (size(targets,2)==1)
            return accuracy(outputs[:,1], targets[:,1]);
        else
            classComparison = targets .== outputs
            correctClassifications = all(classComparison, dims=2)
            return mean(correctClassifications)
        end;
    else
        # Cada patron esta en cada columna
        if (size(targets,1)==1)
            return accuracy(outputs[1,:], targets[1,:]);
        else
            classComparison = targets .== outputs
            correctClassifications = all(classComparison, dims=1)
            return mean(correctClassifications)
        end;
    end;
end;

accuracy(outputs::Array{Float64,1}, targets::Array{Bool,1}; threshold::Float64=0.5) = accuracy(Array{Bool,1}(outputs.>=threshold), targets);
function accuracy(outputs::Array{Float64,2}, targets::Array{Bool,2}; dataInRows::Bool=true)
    @assert(all(size(outputs).==size(targets)));
    if (dataInRows)
        # Cada patron esta en cada fila
        if (size(targets,2)==1)
            return accuracy(outputs[:,1], targets[:,1]);
        else
            return accuracy(classifyOutputs(outputs; dataInRows=true), targets);
        end;
    else
        # Cada patron esta en cada columna
        if (size(targets,1)==1)
            return accuracy(outputs[1,:], targets[1,:]);
        else
            return accuracy(classifyOutputs(outputs; dataInRows=false), targets);
        end;
    end;
end;
accuracy(outputs::Array{Float32,1}, targets::Array{Bool,1}; threshold::Float64=0.5) = accuracy(Float64.(outputs), targets; threshold=threshold);
accuracy(outputs::Array{Float32,2}, targets::Array{Bool,2}; dataInRows::Bool=true)  = accuracy(Float64.(outputs), targets; dataInRows=dataInRows);

calculateMinMaxNormalizationParameters(dataset::Array{Float64,2}; dataInRows=true) =
    ( minimum(dataset, dims=(dataInRows ? 1 : 2)), maximum(dataset, dims=(dataInRows ? 1 : 2)) );

function normalizeMinMax!(dataset::Array{Float64,2}, normalizationParameters::Tuple{Array{Float64,2},Array{Float64,2}}; dataInRows=true)
    min = normalizationParameters[1];
    max = normalizationParameters[2];
    dataset .-= min;
    dataset ./= (max .- min);
    # Si hay algun atributo en el que todos los valores son iguales, se pone a 0
    if (dataInRows)
        dataset[:, vec(min.==max)] .= 0;
    else
        dataset[vec(min.==max), :] .= 0;
    end
end;
normalizeMinMax!(dataset::Array{Float64,2}; dataInRows=true) = normalizeMinMax!(dataset, calculateMinMaxNormalizationParameters(dataset; dataInRows=dataInRows); dataInRows=dataInRows);

function buildClassANN(numInputs::Int64, topology::Array{Int64,1}, numOutputs::Int64)
    ann=Chain();
    numInputsLayer = numInputs;
    for numOutputLayers = topology
        ann = Chain(ann..., Dense(numInputsLayer, numOutputLayers, σ));
        numInputsLayer = numOutputLayers;
    end;
    if (numOutputs == 1)
        ann = Chain(ann..., Dense(numInputsLayer, 1, σ));
    else
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity));
        ann = Chain(ann..., softmax);
    end;
    return ann;
end;

function trainClassANN(topology::Array{Int64,1},
    trainingInputs::Array{Float64,2},   trainingTargets::Array{Bool,2},
    validationInputs::Array{Float64,2}, validationTargets::Array{Bool,2},
    testInputs::Array{Float64,2},       testTargets::Array{Bool,2};
    maxEpochs::Int64=1000, minLoss::Float64=0.0, learningRate::Float64=0.1, maxEpochsVal::Int64=6, showText::Bool=false)

    # Se supone que tenemos cada patron en cada fila
    # Comprobamos que el numero de filas (numero de patrones) coincide tanto en entrenamiento como en validation como test
    @assert(size(trainingInputs,1)==size(trainingTargets,1));
    @assert(size(validationInputs,1)==size(validationTargets,1));
    @assert(size(testInputs,1)==size(testTargets,1));
    # Comprobamos que el numero de columnas coincide en los grupos de entrenamiento, validacion y test
    @assert(size(trainingInputs,2)==size(validationInputs,2)==size(testInputs,2));
    @assert(size(trainingTargets,2)==size(validationTargets,2)==size(testTargets,2));
    # Creamos la RNA
    ann = buildClassANN(size(trainingInputs,2), topology, size(trainingTargets,2));
    # Definimos la funcion de loss
    loss(x,y) = (size(y,1) == 1) ? Flux.Losses.binarycrossentropy(ann(x),y) : Flux.Losses.crossentropy(ann(x),y);
    # Creamos los vectores con los valores de loss y de precision en cada ciclo
    trainingLosses = Float64[];
    trainingAccuracies = Float64[];
    validationLosses = Float64[];
    validationAccuracies = Float64[];
    testLosses = Float64[];
    testAccuracies = Float64[];

    # Empezamos en el ciclo 0
    numEpoch = 0;

    # Una funcion util para calcular los resultados y mostrarlos por pantalla
    function calculateMetrics()
        # Calculamos el loss en entrenamiento y test. Para ello hay que pasar las matrices traspuestas (cada patron en una columna)
        trainingLoss   = loss(trainingInputs',   trainingTargets');
        validationLoss = loss(validationInputs', validationTargets');
        testLoss       = loss(testInputs',       testTargets');
        # Calculamos la salida de la RNA en entrenamiento y test. Para ello hay que pasar la matriz de entradas traspuesta (cada patron en una columna). La matriz de salidas tiene un patron en cada columna
        trainingOutputs   = ann(trainingInputs');
        validationOutputs = ann(validationInputs');
        testOutputs       = ann(testInputs');
        # Para calcular la precision, ponemos 2 opciones aqui equivalentes:
        #  Pasar las matrices con los datos en las columnas. La matriz de salidas ya tiene un patron en cada columna
        trainingAcc   = accuracy(trainingOutputs,   Array{Bool,2}(trainingTargets');   dataInRows=false);
        validationAcc = accuracy(validationOutputs, Array{Bool,2}(validationTargets'); dataInRows=false);
        testAcc       = accuracy(testOutputs,       Array{Bool,2}(testTargets');       dataInRows=false);
        #  Pasar las matrices con los datos en las filas. Hay que trasponer la matriz de salidas de la RNA, puesto que cada dato esta en una fila
        trainingAcc   = accuracy(Array{Float64,2}(trainingOutputs'),   trainingTargets;   dataInRows=true);
        validationAcc = accuracy(Array{Float64,2}(validationOutputs'), validationTargets; dataInRows=true);
        testAcc       = accuracy(Array{Float64,2}(testOutputs'),       testTargets;       dataInRows=true);
        # Mostramos por pantalla el resultado de este ciclo de entrenamiento si nos lo han indicado
        if showText
            println("Epoch ", numEpoch, ": Training loss: ", trainingLoss, ", accuracy: ", 100*trainingAcc, " % - Validation loss: ", validationLoss, ", accuracy: ", 100*validationAcc, " % - Test loss: ", testLoss, ", accuracy: ", 100*testAcc, " %");
        end;
        return (trainingLoss, trainingAcc, validationLoss, validationAcc, testLoss, testAcc)
    end;

    # Calculamos las metricas para el ciclo 0 (sin entrenar nada)
    (trainingLoss, trainingAccuracy, validationLoss, validationAccuracy, testLoss, testAccuracy) = calculateMetrics();
    #  y almacenamos los valores de loss y precision en este ciclo
    push!(trainingLosses,       trainingLoss);
    push!(trainingAccuracies,   trainingAccuracy);
    push!(validationLosses,     validationLoss);
    push!(validationAccuracies, validationAccuracy);
    push!(testLosses,           testLoss);
    push!(testAccuracies,       testAccuracy);

    # Numero de ciclos sin mejorar el error de validacion y el mejor error de validation encontrado hasta el momento
    numEpochsValidation = 0; bestValidationLoss = validationLoss;
    # Cual es la mejor ann que se ha conseguido
    bestANN = deepcopy(ann);

    # Entrenamos hasta que se cumpla una condicion de parada
    while (numEpoch<maxEpochs) && (trainingLoss>minLoss) && (numEpochsValidation<maxEpochsVal)
        # Entrenamos 1 ciclo. Para ello hay que pasar las matrices traspuestas (cada patron en una columna)
        Flux.train!(loss, params(ann), [(trainingInputs', trainingTargets')], ADAM(learningRate));
        # Aumentamos el numero de ciclo en 1
        numEpoch += 1;
        # Calculamos las metricas en este ciclo
        (trainingLoss, trainingAccuracy, validationLoss, validationAccuracy, testLoss, testAccuracy) = calculateMetrics();
        #  y almacenamos los valores de loss y precision en este ciclo
        push!(trainingLosses,       trainingLoss);
        push!(trainingAccuracies,   trainingAccuracy);
        push!(validationLosses,     validationLoss);
        push!(validationAccuracies, validationAccuracy);
        push!(testLosses,           testLoss);
        push!(testAccuracies,       testAccuracy);
        # Aplicamos la parada temprana
        if (validationLoss<bestValidationLoss)
            bestValidationLoss = validationLoss;
            numEpochsValidation = 0;
            bestANN = deepcopy(ann);
        else
            numEpochsValidation += 1;
        end;
    end;
    return (bestANN, trainingLosses, validationLosses, testLosses, trainingAccuracies, validationAccuracies, testAccuracies);
end;

function holdOut(N::Int, P::Float64)
    @assert ((P>=0.) & (P<=1.));
    indices = randperm(N);
    numTrainingInstances = Int(round(N*(1-P)));
    return (indices[1:numTrainingInstances], indices[numTrainingInstances+1:end]);
end;

function holdOut(N::Int, Pval::Float64, Ptest::Float64)
    @assert ((Pval>=0.) & (Pval<=1.));
    @assert ((Ptest>=0.) & (Ptest<=1.));
    @assert ((Pval+Ptest)<=1.);
    # Primero separamos en entrenamiento+validation y test
    (trainingValidationIndices, testIndices) = holdOut(N, Ptest);
    # Después separamos el conjunto de entrenamiento+validation
    (trainingIndices, validationIndices) = holdOut(length(trainingValidationIndices), Pval*N/length(trainingValidationIndices))
    return (trainingValidationIndices[trainingIndices], trainingValidationIndices[validationIndices], testIndices);
end;

function classifyOutputs(outputs::Array{Float64,2}; dataInRows::Bool=true)
    # Miramos donde esta el valor mayor de cada instancia con la funcion findmax
    (_,indicesMaxEachInstance) = findmax(outputs, dims= dataInRows ? 2 : 1);
    # Creamos la matriz de valores booleanos con valores inicialmente a false y asignamos esos indices a true
    outputsBoolean = Array{Bool,2}(falses(size(outputs)));
    outputsBoolean[indicesMaxEachInstance] .= true;
    # Comprobamos que efectivamente cada patron solo este clasificado en una clase
    @assert(all(sum(outputsBoolean, dims=dataInRows ? 2 : 1).==1));
    return outputsBoolean;
end;

function confusionMatrix(outputs::Array{Bool,1}, targets::Array{Bool,1})
    @assert(length(outputs)==length(targets));
    @assert(length(outputs)==length(targets));
    # Para calcular la precision y la tasa de error, se puede llamar a las funciones definidas en la practica 2
    acc         = accuracy(outputs, targets); # Precision, definida previamente en una practica anterior
    errorRate   = 1 - acc;
    recall      = mean(  outputs[  targets]); # Sensibilidad
    specificity = mean(.!outputs[.!targets]); # Especificidad
    precision   = mean(  targets[  outputs]); # Valor predictivo positivo
    NPV         = mean(.!targets[.!outputs]); # Valor predictivo negativo
    # Controlamos que algunos casos pueden ser NaN, y otros no
    @assert(!isnan(recall) && !isnan(specificity));
    precision   = isnan(precision) ? 0 : precision;
    NPV         = isnan(NPV) ? 0 : NPV;
    # Calculamos F1
    F1          = (recall==precision==0.) ? 0. : 2*(recall*precision)/(recall+precision);
    # Reservamos memoria para la matriz de confusion
    confMatrix = Array{Int64,2}(undef, 2, 2);
    # Ponemos en las filas los que pertenecen a cada clase (targets) y en las columnas los clasificados (outputs)
    #  Primera fila/columna: negativos
    #  Segunda fila/columna: positivos
    # Primera fila: patrones de clase negativo, clasificados como negativos o positivos
    confMatrix[1,1] = sum(.!targets .& .!outputs); # VN
    confMatrix[1,2] = sum(.!targets .&   outputs); # FP
    # Segunda fila: patrones de clase positiva, clasificados como negativos o positivos
    confMatrix[2,1] = sum(  targets .& .!outputs); # FN
    confMatrix[2,2] = sum(  targets .&   outputs); # VP
    return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix)
end;

confusionMatrix(outputs::Array{Float64,1}, targets::Array{Bool,1}; threshold::Float64=0.5) = confusionMatrix(Array{Bool,1}(outputs.>=threshold), targets);


function confusionMatrix(outputs::Array{Bool,2}, targets::Array{Bool,2}; weighted::Bool=true)
    @assert(size(outputs)==size(targets));
    numClasses = size(targets,2);
    # Nos aseguramos de que no hay dos columnas
    @assert(numClasses!=2);
    if (numClasses==1)
        return confusionMatrix(outputs[:,1], targets[:,1]);
    else
        # Nos aseguramos de que en cada fila haya uno y sólo un valor a true
        @assert(all(sum(outputs, dims=2).==1));
        # Reservamos memoria para las metricas de cada clase, inicializandolas a 0 porque algunas posiblemente no se calculen
        recall      = zeros(numClasses);
        specificity = zeros(numClasses);
        precision   = zeros(numClasses);
        NPV         = zeros(numClasses);
        F1          = zeros(numClasses);
        # Reservamos memoria para la matriz de confusion
        confMatrix  = Array{Int64,2}(undef, numClasses, numClasses);
        # Calculamos el numero de patrones de cada clase
        numInstancesFromEachClass = vec(sum(targets, dims=1));
        # Calculamos las metricas para cada clase, esto se haria con un bucle similar a "for numClass in 1:numClasses" que itere por todas las clases
        #  Sin embargo, solo hacemos este calculo para las clases que tengan algun patron
        #  Puede ocurrir que alguna clase no tenga patrones como consecuencia de haber dividido de forma aleatoria el conjunto de patrones entrenamiento/test
        #  En aquellas clases en las que no haya patrones, los valores de las metricas seran 0 (los vectores ya estan asignados), y no se tendran en cuenta a la hora de unir estas metricas
        for numClass in findall(numInstancesFromEachClass.>0)
            # Calculamos las metricas de cada problema binario correspondiente a cada clase y las almacenamos en los vectores correspondientes
            (_, _, recall[numClass], specificity[numClass], precision[numClass], NPV[numClass], F1[numClass], _) = confusionMatrix(outputs[:,numClass], targets[:,numClass]);
        end;

        # Reservamos memoria para la matriz de confusion
        confMatrix = Array{Int64,2}(undef, numClasses, numClasses);
        # Calculamos la matriz de confusión haciendo un bucle doble que itere sobre las clases
        for numClassTarget in 1:numClasses, numClassOutput in 1:numClasses
            # Igual que antes, ponemos en las filas los que pertenecen a cada clase (targets) y en las columnas los clasificados (outputs)
            confMatrix[numClassTarget, numClassOutput] = sum(targets[:,numClassTarget] .& outputs[:,numClassOutput]);
        end;

        # Aplicamos las forma de combinar las metricas macro o weighted
        if weighted
            # Calculamos los valores de ponderacion para hacer el promedio
            weights = numInstancesFromEachClass./sum(numInstancesFromEachClass);
            recall      = sum(weights.*recall);
            specificity = sum(weights.*specificity);
            precision   = sum(weights.*precision);
            NPV         = sum(weights.*NPV);
            F1          = sum(weights.*F1);
        else
            # No realizo la media tal cual con la funcion mean, porque puede haber clases sin instancias
            #  En su lugar, realizo la media solamente de las clases que tengan instancias
            numClassesWithInstances = sum(numInstancesFromEachClass.>0);
            recall      = sum(recall)/numClassesWithInstances;
            specificity = sum(specificity)/numClassesWithInstances;
            precision   = sum(precision)/numClassesWithInstances;
            NPV         = sum(NPV)/numClassesWithInstances;
            F1          = sum(F1)/numClassesWithInstances;
        end;
        # Precision y tasa de error las calculamos con las funciones definidas previamente
        acc = accuracy(outputs, targets; dataInRows=true);
        errorRate = 1 - acc;

        return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix);
    end;
end;


function confusionMatrix(outputs::Array{Any,1}, targets::Array{Any,1}; weighted::Bool=true)
    # Comprobamos que todas las clases de salida esten dentro de las clases de las salidas deseadas
    @assert(all([in(output, unique(targets)) for output in outputs]));
    classes = unique(targets);
    # Es importante calcular el vector de clases primero y pasarlo como argumento a las 2 llamadas a oneHotEncoding para que el orden de las clases sea el mismo en ambas matrices
    return confusionMatrix(oneHotEncoding(outputs, classes), oneHotEncoding(targets, classes); weighted=weighted);
end;

confusionMatrix(outputs::Array{Float64,2}, targets::Array{Bool,2}; weighted::Bool=true) = confusionMatrix(classifyOutputs(outputs), targets; weighted=weighted);

# De forma similar a la anterior, añado estas funcion porque las RR.NN.AA. dan la salida como matrices de valores Float32 en lugar de Float64
# Con estas funcion se pueden usar indistintamente matrices de Float32 o Float64
confusionMatrix(outputs::Array{Float32,2}, targets::Array{Bool,2}; weighted::Bool=true) = confusionMatrix(convert(Array{Float64,2}, outputs), targets; weighted=weighted);
printConfusionMatrix(outputs::Array{Float32,2}, targets::Array{Bool,2}; weighted::Bool=true) = printConfusionMatrix(convert(Array{Float64,2}, outputs), targets; weighted=weighted);

# Funciones auxiliares para visualizar por pantalla la matriz de confusion y las metricas que se derivan de ella
function printConfusionMatrix(outputs::Array{Bool,2}, targets::Array{Bool,2}; weighted::Bool=true)
    (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix) = confusionMatrix(outputs, targets; weighted=weighted);
    numClasses = size(confMatrix,1);
    writeHorizontalLine() = (for i in 1:numClasses+1 print("--------") end; println(""); );
    writeHorizontalLine();
    print("\t| ");
    if (numClasses==2)
        println(" - \t + \t|");
    else
        print.("Cl. ", 1:numClasses, "\t| ");
    end;
    println("");
    writeHorizontalLine();
    for numClassTarget in 1:numClasses
        # print.(confMatrix[numClassTarget,:], "\t");
        if (numClasses==2)
            print(numClassTarget == 1 ? " - \t| " : " + \t| ");
        else
            print("Cl. ", numClassTarget, "\t| ");
        end;
        print.(confMatrix[numClassTarget,:], "\t| ");
        println("");
        writeHorizontalLine();
    end;
    println("Accuracy: ", acc);
    println("Error rate: ", errorRate);
    println("Recall: ", recall);
    println("Specificity: ", specificity);
    println("Precision: ", precision);
    println("Negative predictive value: ", NPV);
    println("F1-score: ", F1);
    return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix);
end;
printConfusionMatrix(outputs::Array{Float64,2}, targets::Array{Bool,2}; weighted::Bool=true) =  printConfusionMatrix(classifyOutputs(outputs), targets; weighted=weighted)

function selectBestAnn(topology::Array{Array{Int64,1},1},learningRate::Float64,
                        numMaxEpochs::Int,validationRatio::Float64,
                        testRatio::Float64,maxEpochsVal::Int,timesRepeated::Int)
    for j in 1:length(topology)
        for i in 0:timesRepeated
            #newInputs = normalizeMinMax(inputs);
            normalizeMinMax!(inputs);
            @assert(all(minimum(inputs, dims=1) .== 0));
            @assert(all(maximum(inputs, dims=1) .== 1));

            (trainingIndices, validationIndices, testIndices) = holdOut(size(inputs,1), validationRatio, testRatio);
            if(i==timesRepeated)
                println(trainingIndices)
                println(validationIndices)
                println(testIndices)
            end
            trainingInputs    = inputs[trainingIndices,:];
            validationInputs  = inputs[validationIndices,:];
            testInputs        = inputs[testIndices,:];
            trainingTargets   = targets[trainingIndices,:];
            validationTargets = targets[validationIndices,:];
            testTargets       = targets[testIndices,:];

            (ann, trainingLosses, validationLosses, testLosses, trainingAccuracies) = trainClassANN(topology[j],
                trainingInputs,   trainingTargets,
                validationInputs, validationTargets,
                testInputs,       testTargets;
                maxEpochs=numMaxEpochs, learningRate=learningRate, maxEpochsVal=maxEpochsVal, showText=true);

            weights = BSON.load("myweights.bson")
            params(ann) = weights

            results=plot()
            plot!(results,1:length(trainingLosses),trainingLosses, xaxis="Epoch",yaxis="Loss",title="Losses iteration " * string(i)*" Top: "*string(topology[j]), label="Training")
            plot!(results,1:length(validationLosses),validationLosses, label="Validation")
            plot!(results,1:length(testLosses),testLosses, label="Test")
            display(results)

            trainingOutputs = collect(ann(trainingInputs')');
            printConfusionMatrix(trainingOutputs, trainingTargets; weighted=true);

            if i == 0 && j==0
                global trainingLossesB = trainingLosses;
                global validationLossesB = validationLosses;
                global testLossesB = testLosses;
            end

            if (mean(validationLosses) <= mean(validationLossesB))
                global weights = params(ann)
                local finalann=deepcopy(ann)
                global finalIteration=deepcopy(i);
                global finalTrainingLosses=deepcopy(trainingLosses);
                global finalValidationLosses=deepcopy(validationLosses);
                global finalTestLosses=deepcopy(testLosses);
                global finalTopology=deepcopy(topology[j])
                #using BSON: @save
                BSON.@save "myweights.bson" weights

                trainingLossesB = trainingLosses
                validationLossesB = validationLosses
                testLossesB = testLosses
                println("SAVED WEIGHTS ------------------------------------------------------------------------")
                println("Iteration ",i)
                println(mean(trainingLossesB))
                println(mean(validationLossesB))
                println(mean(testLossesB))
            end
        end
    end;
    return finalann,finalIteration,finalTopology,finalTrainingLosses,finalValidationLosses,finalTestLosses
end;

dataset=loadTrainingDataset();
inputs=dataset[:,1:6];
inputs=Float64.(inputs);
println("Tamaño de la matriz de entradas: ", size(inputs,1), "x", size(inputs,2), " de tipo ", typeof(inputs));
targets = dataset[:,7];
println("Longitud del vector de salidas deseadas antes de codificar: ", length(targets), " de tipo ", typeof(targets));
targets = oneHotEncoding(targets);
println("Tamaño de la matriz de salidas deseadas despues de codificar: ", size(targets,1), "x", size(targets,2), " de tipo ", typeof(targets));
# Comprobamos que ambas matrices tienen el mismo número de filas
@assert (size(inputs,1)==size(targets,1)) "Las matrices de entradas y salidas deseadas no tienen el mismo numero de filas"


topology = [[4, 3],[5,6]];
learningRate = 0.01;
numMaxEpochs = 1000;
validationRatio = 0.2;
testRatio = 0.2;
maxEpochsVal = 100;
timesRepeated = 1;

local ann;

(ann,iteration,finalTopology,trainingLosses,validationLosses,testLosses) = selectBestAnn(topology,learningRate,numMaxEpochs,validationRatio,testRatio,maxEpochsVal,timesRepeated)
finalResults=plot()
plot!(finalResults,1:length(trainingLosses),trainingLosses, xaxis="Epoch",yaxis="Loss",title="Losses Topology: "*string(finalTopology), label="Training")
plot!(finalResults,1:length(validationLosses),validationLosses, label="Validation")
plot!(finalResults,1:length(testLosses),testLosses, label="Test")
display(finalResults)
println(iteration)

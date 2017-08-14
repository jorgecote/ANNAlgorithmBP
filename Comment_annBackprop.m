%*************************************************************************%
%               ANN algoritmo de retro-propagación                        %
%                   - MLP 1 salida                                        %
%                   - Función de activación lineal en la salida           %
%                   - Normaliza valores de entrada y salida               %
%                   - Función de activación sigmoidal en la capa oculta   %
%        Argumentos:                                                      %
%           * traindataset: conjunto de datos de entrenamiento            %
%           * testdataset : conjunto de datos de validación               %
%           * n (eta): tasa de aprendizaje                                %
%           * M      : Numero de neurodos en la capa oculta               %
%           * I      : Numero de iteraciones                              %
%           * alpha  : pendiente de cambio en la función sigmoidal        %
%*************************************************************************%
function [traindataset yHat yHat_val etotal e etotal_val]=annBackprop(traindataset,testdataset,n,M,I,alpha)
%Normalización de conjuntos de entrenamiento y validación
norm_dataset=normaliz(traindataset,'ann-maxmin');
norm_dataset_val=normaliz(testdataset,'ann-maxmin');
datasetsize=size(traindataset);
%Cantidad de muestras
Q=datasetsize(1);
%create dataset input
a_norm=norm_dataset(:,1:size(norm_dataset,2)-1);
%Se agrega terminos de bias
X=[a_norm zeros(Q,1)];   %ones bias term
val_X=[norm_dataset_val(:,1:end-1) ones(size(norm_dataset_val,1),1)];
%Conjunto de salida
Y=traindataset(:,size(traindataset,2));
inputLayerSize=size(X,2);
outputLayerSize=size(Y,2);
hiddenLayerSize=M;
i=1;
etotal(1)=1;
%**********************Retropropagación***********************************%
%inicialización aleatoria de pesos +-0.05
%wnm pesos capa oculta
wnm = -0.05 + (0.05-(-0.05)).*rand(inputLayerSize,hiddenLayerSize);     
%um pesos capa salida
um= (-0.05 + (0.05-(-0.05)).*rand(hiddenLayerSize+1,1));
while(etotal(i)>0.00001)
    i=i+1;
    
    for q=1:size(X,1)
        %feedfordward                        %tamaños de matrices
        z2=X(q,:)*wnm ;                      %z2=>1xM
        a2=[1./(1+exp(-(alpha*z2))) 1];      %a2=>1xM+1
        z3=a2*um  ;                          %z3=>1x1
        yHat(q)=z3  ;                        %yHat=>1x1

        %Backpropagation
        delta_o=yHat(q)-Y(q) ;               %delta_o=>1x1
        gradiente_o=delta_o*a2  ;            %gradiente_o=>1xM

        delta_h=delta_o*um'.*a2.*(1-a2);     %delta_h=>1xM+1
        gradiente_h=(delta_h(1:hiddenLayerSize)'*X(q,:)) ;   %gradiente_h=>MxN
        %Actualizacion de pesos
        wnm=wnm-(n*gradiente_h');
        %traindataset=wnm;
        um=um-(n*gradiente_o');
    end
    %Error acumulado por iteraciones entrenamiento
    etotal(i)=(1/Q)*sum((yHat'-Y).^2);
    etotal(i)
    %Error acumulado de validacion
    for q=1:size(testdataset,1)
        %feedfordward para validación    
        z2_val=val_X(q,:)*wnm ;                       %z2=>1xM
        a2_val=[1./(1+exp(-(alpha*z2_val))) 1];      %a2=>1xM+1
        z3_val=a2_val*um  ;                          %z3=>1x1
        yHat_val(q)=z3_val  ;                        %yHat=>1x1
    end
    etotal_val(i)=(1/size(testdataset,1))*sum((yHat_val'-testdataset(:,end)).^2);
    if(i==I)
           etotal(i)=0; 
        end
end
%error final
e=(yHat'-Y).^2;
end
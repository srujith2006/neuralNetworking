from keras.models import Model
from keras.layers import Input,Flatten,Dense,Concatenate

input_layer=Input(shape=(28,28))

flatten=Flatten(input_shape=(28,28))(input_layer)

h1=Dense(128,activation='relu')(flatten)
h2=Dense(256,activation='relu')(flatten)
h11=Dense(128,activation='relu')(h1)
merge=Concatenate()([h11,h2])

output_layer=Dense(10,activation='softmax')(merge)

model=Model(inputs=input_layer,outputs=output_layer)

model.summary()

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


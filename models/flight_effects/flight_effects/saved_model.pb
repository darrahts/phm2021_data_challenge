’®
Ń¢
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
Į
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ø
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68’¦
p

hd1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:F*
shared_name
hd1/kernel
i
hd1/kernel/Read/ReadVariableOpReadVariableOp
hd1/kernel*
_output_shapes

:F*
dtype0
h
hd1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:F*
shared_name
hd1/bias
a
hd1/bias/Read/ReadVariableOpReadVariableOphd1/bias*
_output_shapes
:F*
dtype0
p

hd2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Fi*
shared_name
hd2/kernel
i
hd2/kernel/Read/ReadVariableOpReadVariableOp
hd2/kernel*
_output_shapes

:Fi*
dtype0
h
hd2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*
shared_name
hd2/bias
a
hd2/bias/Read/ReadVariableOpReadVariableOphd2/bias*
_output_shapes
:i*
dtype0
q

hd3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	i*
shared_name
hd3/kernel
j
hd3/kernel/Read/ReadVariableOpReadVariableOp
hd3/kernel*
_output_shapes
:	i*
dtype0
i
hd3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
hd3/bias
b
hd3/bias/Read/ReadVariableOpReadVariableOphd3/bias*
_output_shapes	
:*
dtype0
q

hd4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	i*
shared_name
hd4/kernel
j
hd4/kernel/Read/ReadVariableOpReadVariableOp
hd4/kernel*
_output_shapes
:	i*
dtype0
h
hd4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*
shared_name
hd4/bias
a
hd4/bias/Read/ReadVariableOpReadVariableOphd4/bias*
_output_shapes
:i*
dtype0
p

hd5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:iF*
shared_name
hd5/kernel
i
hd5/kernel/Read/ReadVariableOpReadVariableOp
hd5/kernel*
_output_shapes

:iF*
dtype0
h
hd5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:F*
shared_name
hd5/bias
a
hd5/bias/Read/ReadVariableOpReadVariableOphd5/bias*
_output_shapes
:F*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:F*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:F*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
`
beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_1
Y
beta_1/Read/ReadVariableOpReadVariableOpbeta_1*
_output_shapes
: *
dtype0
`
beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_2
Y
beta_2/Read/ReadVariableOpReadVariableOpbeta_2*
_output_shapes
: *
dtype0
^
decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecay
W
decay/Read/ReadVariableOpReadVariableOpdecay*
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
~
Adam/hd1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:F*"
shared_nameAdam/hd1/kernel/m
w
%Adam/hd1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/hd1/kernel/m*
_output_shapes

:F*
dtype0
v
Adam/hd1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:F* 
shared_nameAdam/hd1/bias/m
o
#Adam/hd1/bias/m/Read/ReadVariableOpReadVariableOpAdam/hd1/bias/m*
_output_shapes
:F*
dtype0
~
Adam/hd2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Fi*"
shared_nameAdam/hd2/kernel/m
w
%Adam/hd2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/hd2/kernel/m*
_output_shapes

:Fi*
dtype0
v
Adam/hd2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:i* 
shared_nameAdam/hd2/bias/m
o
#Adam/hd2/bias/m/Read/ReadVariableOpReadVariableOpAdam/hd2/bias/m*
_output_shapes
:i*
dtype0

Adam/hd3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	i*"
shared_nameAdam/hd3/kernel/m
x
%Adam/hd3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/hd3/kernel/m*
_output_shapes
:	i*
dtype0
w
Adam/hd3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameAdam/hd3/bias/m
p
#Adam/hd3/bias/m/Read/ReadVariableOpReadVariableOpAdam/hd3/bias/m*
_output_shapes	
:*
dtype0

Adam/hd4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	i*"
shared_nameAdam/hd4/kernel/m
x
%Adam/hd4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/hd4/kernel/m*
_output_shapes
:	i*
dtype0
v
Adam/hd4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:i* 
shared_nameAdam/hd4/bias/m
o
#Adam/hd4/bias/m/Read/ReadVariableOpReadVariableOpAdam/hd4/bias/m*
_output_shapes
:i*
dtype0
~
Adam/hd5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:iF*"
shared_nameAdam/hd5/kernel/m
w
%Adam/hd5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/hd5/kernel/m*
_output_shapes

:iF*
dtype0
v
Adam/hd5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:F* 
shared_nameAdam/hd5/bias/m
o
#Adam/hd5/bias/m/Read/ReadVariableOpReadVariableOpAdam/hd5/bias/m*
_output_shapes
:F*
dtype0

Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:F*$
shared_nameAdam/dense/kernel/m
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

:F*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:*
dtype0
~
Adam/hd1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:F*"
shared_nameAdam/hd1/kernel/v
w
%Adam/hd1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/hd1/kernel/v*
_output_shapes

:F*
dtype0
v
Adam/hd1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:F* 
shared_nameAdam/hd1/bias/v
o
#Adam/hd1/bias/v/Read/ReadVariableOpReadVariableOpAdam/hd1/bias/v*
_output_shapes
:F*
dtype0
~
Adam/hd2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Fi*"
shared_nameAdam/hd2/kernel/v
w
%Adam/hd2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/hd2/kernel/v*
_output_shapes

:Fi*
dtype0
v
Adam/hd2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:i* 
shared_nameAdam/hd2/bias/v
o
#Adam/hd2/bias/v/Read/ReadVariableOpReadVariableOpAdam/hd2/bias/v*
_output_shapes
:i*
dtype0

Adam/hd3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	i*"
shared_nameAdam/hd3/kernel/v
x
%Adam/hd3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/hd3/kernel/v*
_output_shapes
:	i*
dtype0
w
Adam/hd3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameAdam/hd3/bias/v
p
#Adam/hd3/bias/v/Read/ReadVariableOpReadVariableOpAdam/hd3/bias/v*
_output_shapes	
:*
dtype0

Adam/hd4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	i*"
shared_nameAdam/hd4/kernel/v
x
%Adam/hd4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/hd4/kernel/v*
_output_shapes
:	i*
dtype0
v
Adam/hd4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:i* 
shared_nameAdam/hd4/bias/v
o
#Adam/hd4/bias/v/Read/ReadVariableOpReadVariableOpAdam/hd4/bias/v*
_output_shapes
:i*
dtype0
~
Adam/hd5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:iF*"
shared_nameAdam/hd5/kernel/v
w
%Adam/hd5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/hd5/kernel/v*
_output_shapes

:iF*
dtype0
v
Adam/hd5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:F* 
shared_nameAdam/hd5/bias/v
o
#Adam/hd5/bias/v/Read/ReadVariableOpReadVariableOpAdam/hd5/bias/v*
_output_shapes
:F*
dtype0

Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:F*$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes

:F*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
K
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ŹJ
valueĄJB½J B¶J
Ć
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
	optimizer
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
¦

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
¦

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses*
¦

!kernel
"bias
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses*
¦

)kernel
*bias
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses*
¦

1kernel
2bias
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses*
¦

9kernel
:bias
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses*
Ø

Abeta_1

Bbeta_2
	Cdecay
Dlearning_rate
Eitermtmumvmw!mx"my)mz*m{1m|2m}9m~:mvvvv!v"v)v*v1v2v9v:v*
Z
0
1
2
3
!4
"5
)6
*7
18
29
910
:11*
Z
0
1
2
3
!4
"5
)6
*7
18
29
910
:11*
* 
°
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

Kserving_default* 
ZT
VARIABLE_VALUE
hd1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEhd1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

Lnon_trainable_variables

Mlayers
Nmetrics
Olayer_regularization_losses
Player_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
ZT
VARIABLE_VALUE
hd2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEhd2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

Qnon_trainable_variables

Rlayers
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses*
* 
* 
ZT
VARIABLE_VALUE
hd3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEhd3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

!0
"1*

!0
"1*
* 

Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses*
* 
* 
ZT
VARIABLE_VALUE
hd4/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEhd4/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

)0
*1*

)0
*1*
* 

[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses*
* 
* 
ZT
VARIABLE_VALUE
hd5/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEhd5/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

10
21*

10
21*
* 

`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses*
* 
* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

90
:1*

90
:1*
* 

enon_trainable_variables

flayers
gmetrics
hlayer_regularization_losses
ilayer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses*
* 
* 
KE
VARIABLE_VALUEbeta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEbeta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
IC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
* 
5
0
1
2
3
4
5
6*

j0
k1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
	ltotal
	mcount
n	variables
o	keras_api*
8
	ptotal
	qcount
r	variables
s	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

l0
m1*

n	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*

p0
q1*

r	variables*
}w
VARIABLE_VALUEAdam/hd1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/hd1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/hd2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/hd2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/hd3/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/hd3/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/hd4/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/hd4/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/hd5/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/hd5/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/hd1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/hd1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/hd2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/hd2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/hd3/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/hd3/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/hd4/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/hd4/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/hd5/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/hd5/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
v
serving_default_in1Placeholder*'
_output_shapes
:’’’’’’’’’*
dtype0*
shape:’’’’’’’’’
Ń
StatefulPartitionedCallStatefulPartitionedCallserving_default_in1
hd1/kernelhd1/bias
hd2/kernelhd2/bias
hd3/kernelhd3/bias
hd4/kernelhd4/bias
hd5/kernelhd5/biasdense/kernel
dense/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_525647
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ą
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamehd1/kernel/Read/ReadVariableOphd1/bias/Read/ReadVariableOphd2/kernel/Read/ReadVariableOphd2/bias/Read/ReadVariableOphd3/kernel/Read/ReadVariableOphd3/bias/Read/ReadVariableOphd4/kernel/Read/ReadVariableOphd4/bias/Read/ReadVariableOphd5/kernel/Read/ReadVariableOphd5/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpbeta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpAdam/iter/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp%Adam/hd1/kernel/m/Read/ReadVariableOp#Adam/hd1/bias/m/Read/ReadVariableOp%Adam/hd2/kernel/m/Read/ReadVariableOp#Adam/hd2/bias/m/Read/ReadVariableOp%Adam/hd3/kernel/m/Read/ReadVariableOp#Adam/hd3/bias/m/Read/ReadVariableOp%Adam/hd4/kernel/m/Read/ReadVariableOp#Adam/hd4/bias/m/Read/ReadVariableOp%Adam/hd5/kernel/m/Read/ReadVariableOp#Adam/hd5/bias/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp%Adam/hd1/kernel/v/Read/ReadVariableOp#Adam/hd1/bias/v/Read/ReadVariableOp%Adam/hd2/kernel/v/Read/ReadVariableOp#Adam/hd2/bias/v/Read/ReadVariableOp%Adam/hd3/kernel/v/Read/ReadVariableOp#Adam/hd3/bias/v/Read/ReadVariableOp%Adam/hd4/kernel/v/Read/ReadVariableOp#Adam/hd4/bias/v/Read/ReadVariableOp%Adam/hd5/kernel/v/Read/ReadVariableOp#Adam/hd5/bias/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOpConst*:
Tin3
12/	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__traced_save_525924
×
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename
hd1/kernelhd1/bias
hd2/kernelhd2/bias
hd3/kernelhd3/bias
hd4/kernelhd4/bias
hd5/kernelhd5/biasdense/kernel
dense/biasbeta_1beta_2decaylearning_rate	Adam/itertotalcounttotal_1count_1Adam/hd1/kernel/mAdam/hd1/bias/mAdam/hd2/kernel/mAdam/hd2/bias/mAdam/hd3/kernel/mAdam/hd3/bias/mAdam/hd4/kernel/mAdam/hd4/bias/mAdam/hd5/kernel/mAdam/hd5/bias/mAdam/dense/kernel/mAdam/dense/bias/mAdam/hd1/kernel/vAdam/hd1/bias/vAdam/hd2/kernel/vAdam/hd2/bias/vAdam/hd3/kernel/vAdam/hd3/bias/vAdam/hd4/kernel/vAdam/hd4/bias/vAdam/hd5/kernel/vAdam/hd5/bias/vAdam/dense/kernel/vAdam/dense/bias/v*9
Tin2
02.*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__traced_restore_526069Łš
®/
·
A__inference_model_layer_call_and_return_conditional_losses_525571

inputs4
"hd1_matmul_readvariableop_resource:F1
#hd1_biasadd_readvariableop_resource:F4
"hd2_matmul_readvariableop_resource:Fi1
#hd2_biasadd_readvariableop_resource:i5
"hd3_matmul_readvariableop_resource:	i2
#hd3_biasadd_readvariableop_resource:	5
"hd4_matmul_readvariableop_resource:	i1
#hd4_biasadd_readvariableop_resource:i4
"hd5_matmul_readvariableop_resource:iF1
#hd5_biasadd_readvariableop_resource:F6
$dense_matmul_readvariableop_resource:F3
%dense_biasadd_readvariableop_resource:
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢hd1/BiasAdd/ReadVariableOp¢hd1/MatMul/ReadVariableOp¢hd2/BiasAdd/ReadVariableOp¢hd2/MatMul/ReadVariableOp¢hd3/BiasAdd/ReadVariableOp¢hd3/MatMul/ReadVariableOp¢hd4/BiasAdd/ReadVariableOp¢hd4/MatMul/ReadVariableOp¢hd5/BiasAdd/ReadVariableOp¢hd5/MatMul/ReadVariableOp|
hd1/MatMul/ReadVariableOpReadVariableOp"hd1_matmul_readvariableop_resource*
_output_shapes

:F*
dtype0q

hd1/MatMulMatMulinputs!hd1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’Fz
hd1/BiasAdd/ReadVariableOpReadVariableOp#hd1_biasadd_readvariableop_resource*
_output_shapes
:F*
dtype0
hd1/BiasAddBiasAddhd1/MatMul:product:0"hd1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’FX
hd1/ReluReluhd1/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’F|
hd2/MatMul/ReadVariableOpReadVariableOp"hd2_matmul_readvariableop_resource*
_output_shapes

:Fi*
dtype0

hd2/MatMulMatMulhd1/Relu:activations:0!hd2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’iz
hd2/BiasAdd/ReadVariableOpReadVariableOp#hd2_biasadd_readvariableop_resource*
_output_shapes
:i*
dtype0
hd2/BiasAddBiasAddhd2/MatMul:product:0"hd2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’iX
hd2/ReluReluhd2/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’i}
hd3/MatMul/ReadVariableOpReadVariableOp"hd3_matmul_readvariableop_resource*
_output_shapes
:	i*
dtype0

hd3/MatMulMatMulhd2/Relu:activations:0!hd3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’{
hd3/BiasAdd/ReadVariableOpReadVariableOp#hd3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
hd3/BiasAddBiasAddhd3/MatMul:product:0"hd3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’Y
hd3/ReluReluhd3/BiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’}
hd4/MatMul/ReadVariableOpReadVariableOp"hd4_matmul_readvariableop_resource*
_output_shapes
:	i*
dtype0

hd4/MatMulMatMulhd3/Relu:activations:0!hd4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’iz
hd4/BiasAdd/ReadVariableOpReadVariableOp#hd4_biasadd_readvariableop_resource*
_output_shapes
:i*
dtype0
hd4/BiasAddBiasAddhd4/MatMul:product:0"hd4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’iX
hd4/ReluReluhd4/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’i|
hd5/MatMul/ReadVariableOpReadVariableOp"hd5_matmul_readvariableop_resource*
_output_shapes

:iF*
dtype0

hd5/MatMulMatMulhd4/Relu:activations:0!hd5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’Fz
hd5/BiasAdd/ReadVariableOpReadVariableOp#hd5_biasadd_readvariableop_resource*
_output_shapes
:F*
dtype0
hd5/BiasAddBiasAddhd5/MatMul:product:0"hd5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’FX
hd5/ReluReluhd5/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’F
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:F*
dtype0
dense/MatMulMatMulhd5/Relu:activations:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’e
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’ 
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^hd1/BiasAdd/ReadVariableOp^hd1/MatMul/ReadVariableOp^hd2/BiasAdd/ReadVariableOp^hd2/MatMul/ReadVariableOp^hd3/BiasAdd/ReadVariableOp^hd3/MatMul/ReadVariableOp^hd4/BiasAdd/ReadVariableOp^hd4/MatMul/ReadVariableOp^hd5/BiasAdd/ReadVariableOp^hd5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:’’’’’’’’’: : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp28
hd1/BiasAdd/ReadVariableOphd1/BiasAdd/ReadVariableOp26
hd1/MatMul/ReadVariableOphd1/MatMul/ReadVariableOp28
hd2/BiasAdd/ReadVariableOphd2/BiasAdd/ReadVariableOp26
hd2/MatMul/ReadVariableOphd2/MatMul/ReadVariableOp28
hd3/BiasAdd/ReadVariableOphd3/BiasAdd/ReadVariableOp26
hd3/MatMul/ReadVariableOphd3/MatMul/ReadVariableOp28
hd4/BiasAdd/ReadVariableOphd4/BiasAdd/ReadVariableOp26
hd4/MatMul/ReadVariableOphd4/MatMul/ReadVariableOp28
hd5/BiasAdd/ReadVariableOphd5/BiasAdd/ReadVariableOp26
hd5/MatMul/ReadVariableOphd5/MatMul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs


š
?__inference_hd5_layer_call_and_return_conditional_losses_525163

inputs0
matmul_readvariableop_resource:iF-
biasadd_readvariableop_resource:F
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:iF*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’Fr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:F*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’FP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’Fa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’Fw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’i: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’i
 
_user_specified_nameinputs

ķ
A__inference_model_layer_call_and_return_conditional_losses_525186

inputs

hd1_525096:F

hd1_525098:F

hd2_525113:Fi

hd2_525115:i

hd3_525130:	i

hd3_525132:	

hd4_525147:	i

hd4_525149:i

hd5_525164:iF

hd5_525166:F
dense_525180:F
dense_525182:
identity¢dense/StatefulPartitionedCall¢hd1/StatefulPartitionedCall¢hd2/StatefulPartitionedCall¢hd3/StatefulPartitionedCall¢hd4/StatefulPartitionedCall¢hd5/StatefulPartitionedCallß
hd1/StatefulPartitionedCallStatefulPartitionedCallinputs
hd1_525096
hd1_525098*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’F*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_hd1_layer_call_and_return_conditional_losses_525095ż
hd2/StatefulPartitionedCallStatefulPartitionedCall$hd1/StatefulPartitionedCall:output:0
hd2_525113
hd2_525115*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’i*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_hd2_layer_call_and_return_conditional_losses_525112ž
hd3/StatefulPartitionedCallStatefulPartitionedCall$hd2/StatefulPartitionedCall:output:0
hd3_525130
hd3_525132*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_hd3_layer_call_and_return_conditional_losses_525129ż
hd4/StatefulPartitionedCallStatefulPartitionedCall$hd3/StatefulPartitionedCall:output:0
hd4_525147
hd4_525149*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’i*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_hd4_layer_call_and_return_conditional_losses_525146ż
hd5/StatefulPartitionedCallStatefulPartitionedCall$hd4/StatefulPartitionedCall:output:0
hd5_525164
hd5_525166*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’F*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_hd5_layer_call_and_return_conditional_losses_525163
dense/StatefulPartitionedCallStatefulPartitionedCall$hd5/StatefulPartitionedCall:output:0dense_525180dense_525182*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_525179u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’ü
NoOpNoOp^dense/StatefulPartitionedCall^hd1/StatefulPartitionedCall^hd2/StatefulPartitionedCall^hd3/StatefulPartitionedCall^hd4/StatefulPartitionedCall^hd5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:’’’’’’’’’: : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2:
hd1/StatefulPartitionedCallhd1/StatefulPartitionedCall2:
hd2/StatefulPartitionedCallhd2/StatefulPartitionedCall2:
hd3/StatefulPartitionedCallhd3/StatefulPartitionedCall2:
hd4/StatefulPartitionedCallhd4/StatefulPartitionedCall2:
hd5/StatefulPartitionedCallhd5/StatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
¾

$__inference_hd4_layer_call_fn_525716

inputs
unknown:	i
	unknown_0:i
identity¢StatefulPartitionedCall×
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’i*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_hd4_layer_call_and_return_conditional_losses_525146o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’i`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:’’’’’’’’’: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs


š
?__inference_hd5_layer_call_and_return_conditional_losses_525747

inputs0
matmul_readvariableop_resource:iF-
biasadd_readvariableop_resource:F
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:iF*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’Fr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:F*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’FP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’Fa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’Fw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’i: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’i
 
_user_specified_nameinputs
č

„
&__inference_model_layer_call_fn_525497

inputs
unknown:F
	unknown_0:F
	unknown_1:Fi
	unknown_2:i
	unknown_3:	i
	unknown_4:	
	unknown_5:	i
	unknown_6:i
	unknown_7:iF
	unknown_8:F
	unknown_9:F

unknown_10:
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_525186o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:’’’’’’’’’: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs


ń
?__inference_hd4_layer_call_and_return_conditional_losses_525727

inputs1
matmul_readvariableop_resource:	i-
biasadd_readvariableop_resource:i
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	i*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ir
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:i*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’iP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’ia
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’iw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:’’’’’’’’’: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

ķ
A__inference_model_layer_call_and_return_conditional_losses_525338

inputs

hd1_525307:F

hd1_525309:F

hd2_525312:Fi

hd2_525314:i

hd3_525317:	i

hd3_525319:	

hd4_525322:	i

hd4_525324:i

hd5_525327:iF

hd5_525329:F
dense_525332:F
dense_525334:
identity¢dense/StatefulPartitionedCall¢hd1/StatefulPartitionedCall¢hd2/StatefulPartitionedCall¢hd3/StatefulPartitionedCall¢hd4/StatefulPartitionedCall¢hd5/StatefulPartitionedCallß
hd1/StatefulPartitionedCallStatefulPartitionedCallinputs
hd1_525307
hd1_525309*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’F*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_hd1_layer_call_and_return_conditional_losses_525095ż
hd2/StatefulPartitionedCallStatefulPartitionedCall$hd1/StatefulPartitionedCall:output:0
hd2_525312
hd2_525314*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’i*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_hd2_layer_call_and_return_conditional_losses_525112ž
hd3/StatefulPartitionedCallStatefulPartitionedCall$hd2/StatefulPartitionedCall:output:0
hd3_525317
hd3_525319*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_hd3_layer_call_and_return_conditional_losses_525129ż
hd4/StatefulPartitionedCallStatefulPartitionedCall$hd3/StatefulPartitionedCall:output:0
hd4_525322
hd4_525324*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’i*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_hd4_layer_call_and_return_conditional_losses_525146ż
hd5/StatefulPartitionedCallStatefulPartitionedCall$hd4/StatefulPartitionedCall:output:0
hd5_525327
hd5_525329*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’F*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_hd5_layer_call_and_return_conditional_losses_525163
dense/StatefulPartitionedCallStatefulPartitionedCall$hd5/StatefulPartitionedCall:output:0dense_525332dense_525334*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_525179u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’ü
NoOpNoOp^dense/StatefulPartitionedCall^hd1/StatefulPartitionedCall^hd2/StatefulPartitionedCall^hd3/StatefulPartitionedCall^hd4/StatefulPartitionedCall^hd5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:’’’’’’’’’: : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2:
hd1/StatefulPartitionedCallhd1/StatefulPartitionedCall2:
hd2/StatefulPartitionedCallhd2/StatefulPartitionedCall2:
hd3/StatefulPartitionedCallhd3/StatefulPartitionedCall2:
hd4/StatefulPartitionedCallhd4/StatefulPartitionedCall2:
hd5/StatefulPartitionedCallhd5/StatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs


š
?__inference_hd2_layer_call_and_return_conditional_losses_525112

inputs0
matmul_readvariableop_resource:Fi-
biasadd_readvariableop_resource:i
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Fi*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ir
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:i*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’iP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’ia
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’iw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’F: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’F
 
_user_specified_nameinputs
Ä	
ņ
A__inference_dense_layer_call_and_return_conditional_losses_525179

inputs0
matmul_readvariableop_resource:F-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:F*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’F: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’F
 
_user_specified_nameinputs
ģW
š
__inference__traced_save_525924
file_prefix)
%savev2_hd1_kernel_read_readvariableop'
#savev2_hd1_bias_read_readvariableop)
%savev2_hd2_kernel_read_readvariableop'
#savev2_hd2_bias_read_readvariableop)
%savev2_hd3_kernel_read_readvariableop'
#savev2_hd3_bias_read_readvariableop)
%savev2_hd4_kernel_read_readvariableop'
#savev2_hd4_bias_read_readvariableop)
%savev2_hd5_kernel_read_readvariableop'
#savev2_hd5_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop%
!savev2_beta_1_read_readvariableop%
!savev2_beta_2_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop(
$savev2_adam_iter_read_readvariableop	$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop0
,savev2_adam_hd1_kernel_m_read_readvariableop.
*savev2_adam_hd1_bias_m_read_readvariableop0
,savev2_adam_hd2_kernel_m_read_readvariableop.
*savev2_adam_hd2_bias_m_read_readvariableop0
,savev2_adam_hd3_kernel_m_read_readvariableop.
*savev2_adam_hd3_bias_m_read_readvariableop0
,savev2_adam_hd4_kernel_m_read_readvariableop.
*savev2_adam_hd4_bias_m_read_readvariableop0
,savev2_adam_hd5_kernel_m_read_readvariableop.
*savev2_adam_hd5_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop0
,savev2_adam_hd1_kernel_v_read_readvariableop.
*savev2_adam_hd1_bias_v_read_readvariableop0
,savev2_adam_hd2_kernel_v_read_readvariableop.
*savev2_adam_hd2_bias_v_read_readvariableop0
,savev2_adam_hd3_kernel_v_read_readvariableop.
*savev2_adam_hd3_bias_v_read_readvariableop0
,savev2_adam_hd4_kernel_v_read_readvariableop.
*savev2_adam_hd4_bias_v_read_readvariableop0
,savev2_adam_hd5_kernel_v_read_readvariableop.
*savev2_adam_hd5_bias_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: £
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*Ģ
valueĀBæ.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÉ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ­
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0%savev2_hd1_kernel_read_readvariableop#savev2_hd1_bias_read_readvariableop%savev2_hd2_kernel_read_readvariableop#savev2_hd2_bias_read_readvariableop%savev2_hd3_kernel_read_readvariableop#savev2_hd3_bias_read_readvariableop%savev2_hd4_kernel_read_readvariableop#savev2_hd4_bias_read_readvariableop%savev2_hd5_kernel_read_readvariableop#savev2_hd5_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop!savev2_beta_1_read_readvariableop!savev2_beta_2_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop$savev2_adam_iter_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop,savev2_adam_hd1_kernel_m_read_readvariableop*savev2_adam_hd1_bias_m_read_readvariableop,savev2_adam_hd2_kernel_m_read_readvariableop*savev2_adam_hd2_bias_m_read_readvariableop,savev2_adam_hd3_kernel_m_read_readvariableop*savev2_adam_hd3_bias_m_read_readvariableop,savev2_adam_hd4_kernel_m_read_readvariableop*savev2_adam_hd4_bias_m_read_readvariableop,savev2_adam_hd5_kernel_m_read_readvariableop*savev2_adam_hd5_bias_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop,savev2_adam_hd1_kernel_v_read_readvariableop*savev2_adam_hd1_bias_v_read_readvariableop,savev2_adam_hd2_kernel_v_read_readvariableop*savev2_adam_hd2_bias_v_read_readvariableop,savev2_adam_hd3_kernel_v_read_readvariableop*savev2_adam_hd3_bias_v_read_readvariableop,savev2_adam_hd4_kernel_v_read_readvariableop*savev2_adam_hd4_bias_v_read_readvariableop,savev2_adam_hd5_kernel_v_read_readvariableop*savev2_adam_hd5_bias_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *<
dtypes2
02.	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*Ō
_input_shapesĀ
æ: :F:F:Fi:i:	i::	i:i:iF:F:F:: : : : : : : : : :F:F:Fi:i:	i::	i:i:iF:F:F::F:F:Fi:i:	i::	i:i:iF:F:F:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:F: 

_output_shapes
:F:$ 

_output_shapes

:Fi: 

_output_shapes
:i:%!

_output_shapes
:	i:!

_output_shapes	
::%!

_output_shapes
:	i: 

_output_shapes
:i:$	 

_output_shapes

:iF: 


_output_shapes
:F:$ 

_output_shapes

:F: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:F: 

_output_shapes
:F:$ 

_output_shapes

:Fi: 

_output_shapes
:i:%!

_output_shapes
:	i:!

_output_shapes	
::%!

_output_shapes
:	i: 

_output_shapes
:i:$ 

_output_shapes

:iF: 

_output_shapes
:F:$  

_output_shapes

:F: !

_output_shapes
::$" 

_output_shapes

:F: #

_output_shapes
:F:$$ 

_output_shapes

:Fi: %

_output_shapes
:i:%&!

_output_shapes
:	i:!'

_output_shapes	
::%(!

_output_shapes
:	i: )

_output_shapes
:i:$* 

_output_shapes

:iF: +

_output_shapes
:F:$, 

_output_shapes

:F: -

_output_shapes
::.

_output_shapes
: 


š
?__inference_hd1_layer_call_and_return_conditional_losses_525667

inputs0
matmul_readvariableop_resource:F-
biasadd_readvariableop_resource:F
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:F*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’Fr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:F*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’FP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’Fa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’Fw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ß

¢
&__inference_model_layer_call_fn_525394
in1
unknown:F
	unknown_0:F
	unknown_1:Fi
	unknown_2:i
	unknown_3:	i
	unknown_4:	
	unknown_5:	i
	unknown_6:i
	unknown_7:iF
	unknown_8:F
	unknown_9:F

unknown_10:
identity¢StatefulPartitionedCallŁ
StatefulPartitionedCallStatefulPartitionedCallin1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_525338o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:’’’’’’’’’: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
'
_output_shapes
:’’’’’’’’’

_user_specified_namein1


š
?__inference_hd1_layer_call_and_return_conditional_losses_525095

inputs0
matmul_readvariableop_resource:F-
biasadd_readvariableop_resource:F
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:F*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’Fr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:F*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’FP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’Fa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’Fw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs


ń
?__inference_hd4_layer_call_and_return_conditional_losses_525146

inputs1
matmul_readvariableop_resource:	i-
biasadd_readvariableop_resource:i
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	i*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ir
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:i*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’iP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’ia
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’iw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:’’’’’’’’’: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs


š
?__inference_hd2_layer_call_and_return_conditional_losses_525687

inputs0
matmul_readvariableop_resource:Fi-
biasadd_readvariableop_resource:i
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Fi*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ir
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:i*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’iP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’ia
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’iw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’F: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’F
 
_user_specified_nameinputs


ņ
?__inference_hd3_layer_call_and_return_conditional_losses_525129

inputs1
matmul_readvariableop_resource:	i.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	i*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’i: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’i
 
_user_specified_nameinputs
»

$__inference_hd5_layer_call_fn_525736

inputs
unknown:iF
	unknown_0:F
identity¢StatefulPartitionedCall×
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’F*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_hd5_layer_call_and_return_conditional_losses_525163o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’F`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’i: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’i
 
_user_specified_nameinputs
½

 
$__inference_signature_wrapper_525647
in1
unknown:F
	unknown_0:F
	unknown_1:Fi
	unknown_2:i
	unknown_3:	i
	unknown_4:	
	unknown_5:	i
	unknown_6:i
	unknown_7:iF
	unknown_8:F
	unknown_9:F

unknown_10:
identity¢StatefulPartitionedCall¹
StatefulPartitionedCallStatefulPartitionedCallin1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__wrapped_model_525077o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:’’’’’’’’’: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
'
_output_shapes
:’’’’’’’’’

_user_specified_namein1
»

$__inference_hd1_layer_call_fn_525656

inputs
unknown:F
	unknown_0:F
identity¢StatefulPartitionedCall×
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’F*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_hd1_layer_call_and_return_conditional_losses_525095o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’F`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

ź
A__inference_model_layer_call_and_return_conditional_losses_525462
in1

hd1_525431:F

hd1_525433:F

hd2_525436:Fi

hd2_525438:i

hd3_525441:	i

hd3_525443:	

hd4_525446:	i

hd4_525448:i

hd5_525451:iF

hd5_525453:F
dense_525456:F
dense_525458:
identity¢dense/StatefulPartitionedCall¢hd1/StatefulPartitionedCall¢hd2/StatefulPartitionedCall¢hd3/StatefulPartitionedCall¢hd4/StatefulPartitionedCall¢hd5/StatefulPartitionedCallÜ
hd1/StatefulPartitionedCallStatefulPartitionedCallin1
hd1_525431
hd1_525433*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’F*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_hd1_layer_call_and_return_conditional_losses_525095ż
hd2/StatefulPartitionedCallStatefulPartitionedCall$hd1/StatefulPartitionedCall:output:0
hd2_525436
hd2_525438*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’i*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_hd2_layer_call_and_return_conditional_losses_525112ž
hd3/StatefulPartitionedCallStatefulPartitionedCall$hd2/StatefulPartitionedCall:output:0
hd3_525441
hd3_525443*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_hd3_layer_call_and_return_conditional_losses_525129ż
hd4/StatefulPartitionedCallStatefulPartitionedCall$hd3/StatefulPartitionedCall:output:0
hd4_525446
hd4_525448*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’i*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_hd4_layer_call_and_return_conditional_losses_525146ż
hd5/StatefulPartitionedCallStatefulPartitionedCall$hd4/StatefulPartitionedCall:output:0
hd5_525451
hd5_525453*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’F*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_hd5_layer_call_and_return_conditional_losses_525163
dense/StatefulPartitionedCallStatefulPartitionedCall$hd5/StatefulPartitionedCall:output:0dense_525456dense_525458*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_525179u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’ü
NoOpNoOp^dense/StatefulPartitionedCall^hd1/StatefulPartitionedCall^hd2/StatefulPartitionedCall^hd3/StatefulPartitionedCall^hd4/StatefulPartitionedCall^hd5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:’’’’’’’’’: : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2:
hd1/StatefulPartitionedCallhd1/StatefulPartitionedCall2:
hd2/StatefulPartitionedCallhd2/StatefulPartitionedCall2:
hd3/StatefulPartitionedCallhd3/StatefulPartitionedCall2:
hd4/StatefulPartitionedCallhd4/StatefulPartitionedCall2:
hd5/StatefulPartitionedCallhd5/StatefulPartitionedCall:L H
'
_output_shapes
:’’’’’’’’’

_user_specified_namein1

ź
A__inference_model_layer_call_and_return_conditional_losses_525428
in1

hd1_525397:F

hd1_525399:F

hd2_525402:Fi

hd2_525404:i

hd3_525407:	i

hd3_525409:	

hd4_525412:	i

hd4_525414:i

hd5_525417:iF

hd5_525419:F
dense_525422:F
dense_525424:
identity¢dense/StatefulPartitionedCall¢hd1/StatefulPartitionedCall¢hd2/StatefulPartitionedCall¢hd3/StatefulPartitionedCall¢hd4/StatefulPartitionedCall¢hd5/StatefulPartitionedCallÜ
hd1/StatefulPartitionedCallStatefulPartitionedCallin1
hd1_525397
hd1_525399*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’F*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_hd1_layer_call_and_return_conditional_losses_525095ż
hd2/StatefulPartitionedCallStatefulPartitionedCall$hd1/StatefulPartitionedCall:output:0
hd2_525402
hd2_525404*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’i*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_hd2_layer_call_and_return_conditional_losses_525112ž
hd3/StatefulPartitionedCallStatefulPartitionedCall$hd2/StatefulPartitionedCall:output:0
hd3_525407
hd3_525409*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_hd3_layer_call_and_return_conditional_losses_525129ż
hd4/StatefulPartitionedCallStatefulPartitionedCall$hd3/StatefulPartitionedCall:output:0
hd4_525412
hd4_525414*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’i*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_hd4_layer_call_and_return_conditional_losses_525146ż
hd5/StatefulPartitionedCallStatefulPartitionedCall$hd4/StatefulPartitionedCall:output:0
hd5_525417
hd5_525419*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’F*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_hd5_layer_call_and_return_conditional_losses_525163
dense/StatefulPartitionedCallStatefulPartitionedCall$hd5/StatefulPartitionedCall:output:0dense_525422dense_525424*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_525179u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’ü
NoOpNoOp^dense/StatefulPartitionedCall^hd1/StatefulPartitionedCall^hd2/StatefulPartitionedCall^hd3/StatefulPartitionedCall^hd4/StatefulPartitionedCall^hd5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:’’’’’’’’’: : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2:
hd1/StatefulPartitionedCallhd1/StatefulPartitionedCall2:
hd2/StatefulPartitionedCallhd2/StatefulPartitionedCall2:
hd3/StatefulPartitionedCallhd3/StatefulPartitionedCall2:
hd4/StatefulPartitionedCallhd4/StatefulPartitionedCall2:
hd5/StatefulPartitionedCallhd5/StatefulPartitionedCall:L H
'
_output_shapes
:’’’’’’’’’

_user_specified_namein1
æ

&__inference_dense_layer_call_fn_525756

inputs
unknown:F
	unknown_0:
identity¢StatefulPartitionedCallŁ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_525179o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’F: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’F
 
_user_specified_nameinputs
æ

$__inference_hd3_layer_call_fn_525696

inputs
unknown:	i
	unknown_0:	
identity¢StatefulPartitionedCallŲ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_hd3_layer_call_and_return_conditional_losses_525129p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’i: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’i
 
_user_specified_nameinputs
č

„
&__inference_model_layer_call_fn_525526

inputs
unknown:F
	unknown_0:F
	unknown_1:Fi
	unknown_2:i
	unknown_3:	i
	unknown_4:	
	unknown_5:	i
	unknown_6:i
	unknown_7:iF
	unknown_8:F
	unknown_9:F

unknown_10:
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_525338o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:’’’’’’’’’: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
®/
·
A__inference_model_layer_call_and_return_conditional_losses_525616

inputs4
"hd1_matmul_readvariableop_resource:F1
#hd1_biasadd_readvariableop_resource:F4
"hd2_matmul_readvariableop_resource:Fi1
#hd2_biasadd_readvariableop_resource:i5
"hd3_matmul_readvariableop_resource:	i2
#hd3_biasadd_readvariableop_resource:	5
"hd4_matmul_readvariableop_resource:	i1
#hd4_biasadd_readvariableop_resource:i4
"hd5_matmul_readvariableop_resource:iF1
#hd5_biasadd_readvariableop_resource:F6
$dense_matmul_readvariableop_resource:F3
%dense_biasadd_readvariableop_resource:
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢hd1/BiasAdd/ReadVariableOp¢hd1/MatMul/ReadVariableOp¢hd2/BiasAdd/ReadVariableOp¢hd2/MatMul/ReadVariableOp¢hd3/BiasAdd/ReadVariableOp¢hd3/MatMul/ReadVariableOp¢hd4/BiasAdd/ReadVariableOp¢hd4/MatMul/ReadVariableOp¢hd5/BiasAdd/ReadVariableOp¢hd5/MatMul/ReadVariableOp|
hd1/MatMul/ReadVariableOpReadVariableOp"hd1_matmul_readvariableop_resource*
_output_shapes

:F*
dtype0q

hd1/MatMulMatMulinputs!hd1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’Fz
hd1/BiasAdd/ReadVariableOpReadVariableOp#hd1_biasadd_readvariableop_resource*
_output_shapes
:F*
dtype0
hd1/BiasAddBiasAddhd1/MatMul:product:0"hd1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’FX
hd1/ReluReluhd1/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’F|
hd2/MatMul/ReadVariableOpReadVariableOp"hd2_matmul_readvariableop_resource*
_output_shapes

:Fi*
dtype0

hd2/MatMulMatMulhd1/Relu:activations:0!hd2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’iz
hd2/BiasAdd/ReadVariableOpReadVariableOp#hd2_biasadd_readvariableop_resource*
_output_shapes
:i*
dtype0
hd2/BiasAddBiasAddhd2/MatMul:product:0"hd2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’iX
hd2/ReluReluhd2/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’i}
hd3/MatMul/ReadVariableOpReadVariableOp"hd3_matmul_readvariableop_resource*
_output_shapes
:	i*
dtype0

hd3/MatMulMatMulhd2/Relu:activations:0!hd3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’{
hd3/BiasAdd/ReadVariableOpReadVariableOp#hd3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
hd3/BiasAddBiasAddhd3/MatMul:product:0"hd3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’Y
hd3/ReluReluhd3/BiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’}
hd4/MatMul/ReadVariableOpReadVariableOp"hd4_matmul_readvariableop_resource*
_output_shapes
:	i*
dtype0

hd4/MatMulMatMulhd3/Relu:activations:0!hd4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’iz
hd4/BiasAdd/ReadVariableOpReadVariableOp#hd4_biasadd_readvariableop_resource*
_output_shapes
:i*
dtype0
hd4/BiasAddBiasAddhd4/MatMul:product:0"hd4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’iX
hd4/ReluReluhd4/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’i|
hd5/MatMul/ReadVariableOpReadVariableOp"hd5_matmul_readvariableop_resource*
_output_shapes

:iF*
dtype0

hd5/MatMulMatMulhd4/Relu:activations:0!hd5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’Fz
hd5/BiasAdd/ReadVariableOpReadVariableOp#hd5_biasadd_readvariableop_resource*
_output_shapes
:F*
dtype0
hd5/BiasAddBiasAddhd5/MatMul:product:0"hd5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’FX
hd5/ReluReluhd5/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’F
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:F*
dtype0
dense/MatMulMatMulhd5/Relu:activations:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’e
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’ 
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^hd1/BiasAdd/ReadVariableOp^hd1/MatMul/ReadVariableOp^hd2/BiasAdd/ReadVariableOp^hd2/MatMul/ReadVariableOp^hd3/BiasAdd/ReadVariableOp^hd3/MatMul/ReadVariableOp^hd4/BiasAdd/ReadVariableOp^hd4/MatMul/ReadVariableOp^hd5/BiasAdd/ReadVariableOp^hd5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:’’’’’’’’’: : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp28
hd1/BiasAdd/ReadVariableOphd1/BiasAdd/ReadVariableOp26
hd1/MatMul/ReadVariableOphd1/MatMul/ReadVariableOp28
hd2/BiasAdd/ReadVariableOphd2/BiasAdd/ReadVariableOp26
hd2/MatMul/ReadVariableOphd2/MatMul/ReadVariableOp28
hd3/BiasAdd/ReadVariableOphd3/BiasAdd/ReadVariableOp26
hd3/MatMul/ReadVariableOphd3/MatMul/ReadVariableOp28
hd4/BiasAdd/ReadVariableOphd4/BiasAdd/ReadVariableOp26
hd4/MatMul/ReadVariableOphd4/MatMul/ReadVariableOp28
hd5/BiasAdd/ReadVariableOphd5/BiasAdd/ReadVariableOp26
hd5/MatMul/ReadVariableOphd5/MatMul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ß

¢
&__inference_model_layer_call_fn_525213
in1
unknown:F
	unknown_0:F
	unknown_1:Fi
	unknown_2:i
	unknown_3:	i
	unknown_4:	
	unknown_5:	i
	unknown_6:i
	unknown_7:iF
	unknown_8:F
	unknown_9:F

unknown_10:
identity¢StatefulPartitionedCallŁ
StatefulPartitionedCallStatefulPartitionedCallin1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_525186o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:’’’’’’’’’: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
'
_output_shapes
:’’’’’’’’’

_user_specified_namein1


ņ
?__inference_hd3_layer_call_and_return_conditional_losses_525707

inputs1
matmul_readvariableop_resource:	i.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	i*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’i: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’i
 
_user_specified_nameinputs
»

$__inference_hd2_layer_call_fn_525676

inputs
unknown:Fi
	unknown_0:i
identity¢StatefulPartitionedCall×
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’i*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_hd2_layer_call_and_return_conditional_losses_525112o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’i`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’F: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’F
 
_user_specified_nameinputs
Ä	
ņ
A__inference_dense_layer_call_and_return_conditional_losses_525766

inputs0
matmul_readvariableop_resource:F-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:F*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’F: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’F
 
_user_specified_nameinputs
5
¤	
!__inference__wrapped_model_525077
in1:
(model_hd1_matmul_readvariableop_resource:F7
)model_hd1_biasadd_readvariableop_resource:F:
(model_hd2_matmul_readvariableop_resource:Fi7
)model_hd2_biasadd_readvariableop_resource:i;
(model_hd3_matmul_readvariableop_resource:	i8
)model_hd3_biasadd_readvariableop_resource:	;
(model_hd4_matmul_readvariableop_resource:	i7
)model_hd4_biasadd_readvariableop_resource:i:
(model_hd5_matmul_readvariableop_resource:iF7
)model_hd5_biasadd_readvariableop_resource:F<
*model_dense_matmul_readvariableop_resource:F9
+model_dense_biasadd_readvariableop_resource:
identity¢"model/dense/BiasAdd/ReadVariableOp¢!model/dense/MatMul/ReadVariableOp¢ model/hd1/BiasAdd/ReadVariableOp¢model/hd1/MatMul/ReadVariableOp¢ model/hd2/BiasAdd/ReadVariableOp¢model/hd2/MatMul/ReadVariableOp¢ model/hd3/BiasAdd/ReadVariableOp¢model/hd3/MatMul/ReadVariableOp¢ model/hd4/BiasAdd/ReadVariableOp¢model/hd4/MatMul/ReadVariableOp¢ model/hd5/BiasAdd/ReadVariableOp¢model/hd5/MatMul/ReadVariableOp
model/hd1/MatMul/ReadVariableOpReadVariableOp(model_hd1_matmul_readvariableop_resource*
_output_shapes

:F*
dtype0z
model/hd1/MatMulMatMulin1'model/hd1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’F
 model/hd1/BiasAdd/ReadVariableOpReadVariableOp)model_hd1_biasadd_readvariableop_resource*
_output_shapes
:F*
dtype0
model/hd1/BiasAddBiasAddmodel/hd1/MatMul:product:0(model/hd1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’Fd
model/hd1/ReluRelumodel/hd1/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’F
model/hd2/MatMul/ReadVariableOpReadVariableOp(model_hd2_matmul_readvariableop_resource*
_output_shapes

:Fi*
dtype0
model/hd2/MatMulMatMulmodel/hd1/Relu:activations:0'model/hd2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’i
 model/hd2/BiasAdd/ReadVariableOpReadVariableOp)model_hd2_biasadd_readvariableop_resource*
_output_shapes
:i*
dtype0
model/hd2/BiasAddBiasAddmodel/hd2/MatMul:product:0(model/hd2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’id
model/hd2/ReluRelumodel/hd2/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’i
model/hd3/MatMul/ReadVariableOpReadVariableOp(model_hd3_matmul_readvariableop_resource*
_output_shapes
:	i*
dtype0
model/hd3/MatMulMatMulmodel/hd2/Relu:activations:0'model/hd3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’
 model/hd3/BiasAdd/ReadVariableOpReadVariableOp)model_hd3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
model/hd3/BiasAddBiasAddmodel/hd3/MatMul:product:0(model/hd3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’e
model/hd3/ReluRelumodel/hd3/BiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’
model/hd4/MatMul/ReadVariableOpReadVariableOp(model_hd4_matmul_readvariableop_resource*
_output_shapes
:	i*
dtype0
model/hd4/MatMulMatMulmodel/hd3/Relu:activations:0'model/hd4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’i
 model/hd4/BiasAdd/ReadVariableOpReadVariableOp)model_hd4_biasadd_readvariableop_resource*
_output_shapes
:i*
dtype0
model/hd4/BiasAddBiasAddmodel/hd4/MatMul:product:0(model/hd4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’id
model/hd4/ReluRelumodel/hd4/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’i
model/hd5/MatMul/ReadVariableOpReadVariableOp(model_hd5_matmul_readvariableop_resource*
_output_shapes

:iF*
dtype0
model/hd5/MatMulMatMulmodel/hd4/Relu:activations:0'model/hd5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’F
 model/hd5/BiasAdd/ReadVariableOpReadVariableOp)model_hd5_biasadd_readvariableop_resource*
_output_shapes
:F*
dtype0
model/hd5/BiasAddBiasAddmodel/hd5/MatMul:product:0(model/hd5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’Fd
model/hd5/ReluRelumodel/hd5/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’F
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes

:F*
dtype0
model/dense/MatMulMatMulmodel/hd5/Relu:activations:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’k
IdentityIdentitymodel/dense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’č
NoOpNoOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp!^model/hd1/BiasAdd/ReadVariableOp ^model/hd1/MatMul/ReadVariableOp!^model/hd2/BiasAdd/ReadVariableOp ^model/hd2/MatMul/ReadVariableOp!^model/hd3/BiasAdd/ReadVariableOp ^model/hd3/MatMul/ReadVariableOp!^model/hd4/BiasAdd/ReadVariableOp ^model/hd4/MatMul/ReadVariableOp!^model/hd5/BiasAdd/ReadVariableOp ^model/hd5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:’’’’’’’’’: : : : : : : : : : : : 2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2D
 model/hd1/BiasAdd/ReadVariableOp model/hd1/BiasAdd/ReadVariableOp2B
model/hd1/MatMul/ReadVariableOpmodel/hd1/MatMul/ReadVariableOp2D
 model/hd2/BiasAdd/ReadVariableOp model/hd2/BiasAdd/ReadVariableOp2B
model/hd2/MatMul/ReadVariableOpmodel/hd2/MatMul/ReadVariableOp2D
 model/hd3/BiasAdd/ReadVariableOp model/hd3/BiasAdd/ReadVariableOp2B
model/hd3/MatMul/ReadVariableOpmodel/hd3/MatMul/ReadVariableOp2D
 model/hd4/BiasAdd/ReadVariableOp model/hd4/BiasAdd/ReadVariableOp2B
model/hd4/MatMul/ReadVariableOpmodel/hd4/MatMul/ReadVariableOp2D
 model/hd5/BiasAdd/ReadVariableOp model/hd5/BiasAdd/ReadVariableOp2B
model/hd5/MatMul/ReadVariableOpmodel/hd5/MatMul/ReadVariableOp:L H
'
_output_shapes
:’’’’’’’’’

_user_specified_namein1
°
Ē
"__inference__traced_restore_526069
file_prefix-
assignvariableop_hd1_kernel:F)
assignvariableop_1_hd1_bias:F/
assignvariableop_2_hd2_kernel:Fi)
assignvariableop_3_hd2_bias:i0
assignvariableop_4_hd3_kernel:	i*
assignvariableop_5_hd3_bias:	0
assignvariableop_6_hd4_kernel:	i)
assignvariableop_7_hd4_bias:i/
assignvariableop_8_hd5_kernel:iF)
assignvariableop_9_hd5_bias:F2
 assignvariableop_10_dense_kernel:F,
assignvariableop_11_dense_bias:$
assignvariableop_12_beta_1: $
assignvariableop_13_beta_2: #
assignvariableop_14_decay: +
!assignvariableop_15_learning_rate: '
assignvariableop_16_adam_iter:	 #
assignvariableop_17_total: #
assignvariableop_18_count: %
assignvariableop_19_total_1: %
assignvariableop_20_count_1: 7
%assignvariableop_21_adam_hd1_kernel_m:F1
#assignvariableop_22_adam_hd1_bias_m:F7
%assignvariableop_23_adam_hd2_kernel_m:Fi1
#assignvariableop_24_adam_hd2_bias_m:i8
%assignvariableop_25_adam_hd3_kernel_m:	i2
#assignvariableop_26_adam_hd3_bias_m:	8
%assignvariableop_27_adam_hd4_kernel_m:	i1
#assignvariableop_28_adam_hd4_bias_m:i7
%assignvariableop_29_adam_hd5_kernel_m:iF1
#assignvariableop_30_adam_hd5_bias_m:F9
'assignvariableop_31_adam_dense_kernel_m:F3
%assignvariableop_32_adam_dense_bias_m:7
%assignvariableop_33_adam_hd1_kernel_v:F1
#assignvariableop_34_adam_hd1_bias_v:F7
%assignvariableop_35_adam_hd2_kernel_v:Fi1
#assignvariableop_36_adam_hd2_bias_v:i8
%assignvariableop_37_adam_hd3_kernel_v:	i2
#assignvariableop_38_adam_hd3_bias_v:	8
%assignvariableop_39_adam_hd4_kernel_v:	i1
#assignvariableop_40_adam_hd4_bias_v:i7
%assignvariableop_41_adam_hd5_kernel_v:iF1
#assignvariableop_42_adam_hd5_bias_v:F9
'assignvariableop_43_adam_dense_kernel_v:F3
%assignvariableop_44_adam_dense_bias_v:
identity_46¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9¦
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*Ģ
valueĀBæ.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHĢ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ī
_output_shapes»
ø::::::::::::::::::::::::::::::::::::::::::::::*<
dtypes2
02.	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_hd1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_hd1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOpassignvariableop_2_hd2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_hd2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOpassignvariableop_4_hd3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_hd3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_hd4_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_hd4_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_hd5_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_hd5_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp assignvariableop_10_dense_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_dense_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_beta_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_beta_2Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_decayIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp!assignvariableop_15_learning_rateIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_iterIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp%assignvariableop_21_adam_hd1_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp#assignvariableop_22_adam_hd1_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp%assignvariableop_23_adam_hd2_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp#assignvariableop_24_adam_hd2_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp%assignvariableop_25_adam_hd3_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp#assignvariableop_26_adam_hd3_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp%assignvariableop_27_adam_hd4_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp#assignvariableop_28_adam_hd4_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp%assignvariableop_29_adam_hd5_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp#assignvariableop_30_adam_hd5_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp'assignvariableop_31_adam_dense_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp%assignvariableop_32_adam_dense_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp%assignvariableop_33_adam_hd1_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp#assignvariableop_34_adam_hd1_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp%assignvariableop_35_adam_hd2_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_36AssignVariableOp#assignvariableop_36_adam_hd2_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_37AssignVariableOp%assignvariableop_37_adam_hd3_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_38AssignVariableOp#assignvariableop_38_adam_hd3_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOp%assignvariableop_39_adam_hd4_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOp#assignvariableop_40_adam_hd4_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_41AssignVariableOp%assignvariableop_41_adam_hd5_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_42AssignVariableOp#assignvariableop_42_adam_hd5_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_43AssignVariableOp'assignvariableop_43_adam_dense_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_44AssignVariableOp%assignvariableop_44_adam_dense_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ­
Identity_45Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_46IdentityIdentity_45:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_46Identity_46:output:0*o
_input_shapes^
\: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"ŪL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp* 
serving_default
3
in1,
serving_default_in1:0’’’’’’’’’9
dense0
StatefulPartitionedCall:0’’’’’’’’’tensorflow/serving/predict:Ów
Ś
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
	optimizer
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
»

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
»

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses"
_tf_keras_layer
»

!kernel
"bias
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses"
_tf_keras_layer
»

)kernel
*bias
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses"
_tf_keras_layer
»

1kernel
2bias
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses"
_tf_keras_layer
»

9kernel
:bias
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses"
_tf_keras_layer
·

Abeta_1

Bbeta_2
	Cdecay
Dlearning_rate
Eitermtmumvmw!mx"my)mz*m{1m|2m}9m~:mvvvv!v"v)v*v1v2v9v:v"
	optimizer
v
0
1
2
3
!4
"5
)6
*7
18
29
910
:11"
trackable_list_wrapper
v
0
1
2
3
!4
"5
)6
*7
18
29
910
:11"
trackable_list_wrapper
 "
trackable_list_wrapper
Ź
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ę2ć
&__inference_model_layer_call_fn_525213
&__inference_model_layer_call_fn_525497
&__inference_model_layer_call_fn_525526
&__inference_model_layer_call_fn_525394Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
Ņ2Ļ
A__inference_model_layer_call_and_return_conditional_losses_525571
A__inference_model_layer_call_and_return_conditional_losses_525616
A__inference_model_layer_call_and_return_conditional_losses_525428
A__inference_model_layer_call_and_return_conditional_losses_525462Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
ČBÅ
!__inference__wrapped_model_525077in1"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
,
Kserving_default"
signature_map
:F2
hd1/kernel
:F2hd1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Lnon_trainable_variables

Mlayers
Nmetrics
Olayer_regularization_losses
Player_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ī2Ė
$__inference_hd1_layer_call_fn_525656¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
é2ę
?__inference_hd1_layer_call_and_return_conditional_losses_525667¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
:Fi2
hd2/kernel
:i2hd2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Qnon_trainable_variables

Rlayers
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses"
_generic_user_object
Ī2Ė
$__inference_hd2_layer_call_fn_525676¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
é2ę
?__inference_hd2_layer_call_and_return_conditional_losses_525687¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
:	i2
hd3/kernel
:2hd3/bias
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
Ī2Ė
$__inference_hd3_layer_call_fn_525696¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
é2ę
?__inference_hd3_layer_call_and_return_conditional_losses_525707¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
:	i2
hd4/kernel
:i2hd4/bias
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
Ī2Ė
$__inference_hd4_layer_call_fn_525716¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
é2ę
?__inference_hd4_layer_call_and_return_conditional_losses_525727¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
:iF2
hd5/kernel
:F2hd5/bias
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
­
`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
Ī2Ė
$__inference_hd5_layer_call_fn_525736¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
é2ę
?__inference_hd5_layer_call_and_return_conditional_losses_525747¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
:F2dense/kernel
:2
dense/bias
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
enon_trainable_variables

flayers
gmetrics
hlayer_regularization_losses
ilayer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
Š2Ķ
&__inference_dense_layer_call_fn_525756¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ė2č
A__inference_dense_layer_call_and_return_conditional_losses_525766¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
: (2beta_1
: (2beta_2
: (2decay
: (2learning_rate
:	 (2	Adam/iter
 "
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ĒBÄ
$__inference_signature_wrapper_525647in1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	ltotal
	mcount
n	variables
o	keras_api"
_tf_keras_metric
N
	ptotal
	qcount
r	variables
s	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
l0
m1"
trackable_list_wrapper
-
n	variables"
_generic_user_object
:  (2total
:  (2count
.
p0
q1"
trackable_list_wrapper
-
r	variables"
_generic_user_object
!:F2Adam/hd1/kernel/m
:F2Adam/hd1/bias/m
!:Fi2Adam/hd2/kernel/m
:i2Adam/hd2/bias/m
": 	i2Adam/hd3/kernel/m
:2Adam/hd3/bias/m
": 	i2Adam/hd4/kernel/m
:i2Adam/hd4/bias/m
!:iF2Adam/hd5/kernel/m
:F2Adam/hd5/bias/m
#:!F2Adam/dense/kernel/m
:2Adam/dense/bias/m
!:F2Adam/hd1/kernel/v
:F2Adam/hd1/bias/v
!:Fi2Adam/hd2/kernel/v
:i2Adam/hd2/bias/v
": 	i2Adam/hd3/kernel/v
:2Adam/hd3/bias/v
": 	i2Adam/hd4/kernel/v
:i2Adam/hd4/bias/v
!:iF2Adam/hd5/kernel/v
:F2Adam/hd5/bias/v
#:!F2Adam/dense/kernel/v
:2Adam/dense/bias/v
!__inference__wrapped_model_525077k!")*129:,¢)
"¢

in1’’’’’’’’’
Ŗ "-Ŗ*
(
dense
dense’’’’’’’’’”
A__inference_dense_layer_call_and_return_conditional_losses_525766\9:/¢,
%¢"
 
inputs’’’’’’’’’F
Ŗ "%¢"

0’’’’’’’’’
 y
&__inference_dense_layer_call_fn_525756O9:/¢,
%¢"
 
inputs’’’’’’’’’F
Ŗ "’’’’’’’’’
?__inference_hd1_layer_call_and_return_conditional_losses_525667\/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ "%¢"

0’’’’’’’’’F
 w
$__inference_hd1_layer_call_fn_525656O/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ "’’’’’’’’’F
?__inference_hd2_layer_call_and_return_conditional_losses_525687\/¢,
%¢"
 
inputs’’’’’’’’’F
Ŗ "%¢"

0’’’’’’’’’i
 w
$__inference_hd2_layer_call_fn_525676O/¢,
%¢"
 
inputs’’’’’’’’’F
Ŗ "’’’’’’’’’i 
?__inference_hd3_layer_call_and_return_conditional_losses_525707]!"/¢,
%¢"
 
inputs’’’’’’’’’i
Ŗ "&¢#

0’’’’’’’’’
 x
$__inference_hd3_layer_call_fn_525696P!"/¢,
%¢"
 
inputs’’’’’’’’’i
Ŗ "’’’’’’’’’ 
?__inference_hd4_layer_call_and_return_conditional_losses_525727])*0¢-
&¢#
!
inputs’’’’’’’’’
Ŗ "%¢"

0’’’’’’’’’i
 x
$__inference_hd4_layer_call_fn_525716P)*0¢-
&¢#
!
inputs’’’’’’’’’
Ŗ "’’’’’’’’’i
?__inference_hd5_layer_call_and_return_conditional_losses_525747\12/¢,
%¢"
 
inputs’’’’’’’’’i
Ŗ "%¢"

0’’’’’’’’’F
 w
$__inference_hd5_layer_call_fn_525736O12/¢,
%¢"
 
inputs’’’’’’’’’i
Ŗ "’’’’’’’’’F°
A__inference_model_layer_call_and_return_conditional_losses_525428k!")*129:4¢1
*¢'

in1’’’’’’’’’
p 

 
Ŗ "%¢"

0’’’’’’’’’
 °
A__inference_model_layer_call_and_return_conditional_losses_525462k!")*129:4¢1
*¢'

in1’’’’’’’’’
p

 
Ŗ "%¢"

0’’’’’’’’’
 ³
A__inference_model_layer_call_and_return_conditional_losses_525571n!")*129:7¢4
-¢*
 
inputs’’’’’’’’’
p 

 
Ŗ "%¢"

0’’’’’’’’’
 ³
A__inference_model_layer_call_and_return_conditional_losses_525616n!")*129:7¢4
-¢*
 
inputs’’’’’’’’’
p

 
Ŗ "%¢"

0’’’’’’’’’
 
&__inference_model_layer_call_fn_525213^!")*129:4¢1
*¢'

in1’’’’’’’’’
p 

 
Ŗ "’’’’’’’’’
&__inference_model_layer_call_fn_525394^!")*129:4¢1
*¢'

in1’’’’’’’’’
p

 
Ŗ "’’’’’’’’’
&__inference_model_layer_call_fn_525497a!")*129:7¢4
-¢*
 
inputs’’’’’’’’’
p 

 
Ŗ "’’’’’’’’’
&__inference_model_layer_call_fn_525526a!")*129:7¢4
-¢*
 
inputs’’’’’’’’’
p

 
Ŗ "’’’’’’’’’
$__inference_signature_wrapper_525647r!")*129:3¢0
¢ 
)Ŗ&
$
in1
in1’’’’’’’’’"-Ŗ*
(
dense
dense’’’’’’’’’
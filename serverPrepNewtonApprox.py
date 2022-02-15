import tenseal as ts
import base64
import scipy.io
from math import sqrt

def write_data(file_name, data):
    if type(data) == bytes:
        #bytes to base64
        data = base64.b64encode(data)

    with open(file_name, 'wb') as f:
        f.write(data)

def read_data(file_name):
    with open(file_name, "rb") as f:
        data = f.read()
    #base64 to bytes
    return base64.b64decode(data)


# Load the mat files

# import channel compensation matrix V from Matlab
# dim V (200,200)
V = scipy.io.loadmat('V.mat')
V=V.get('V')
V=V*10;

# import test iVectors from matlab mat files
# dim (200,302)
test_ivs=scipy.io.loadmat('test_ivs.mat')
T=test_ivs.get('test_ivs')
T=T*10;

# import template iVectors from matlab mat files
# dim (200,151)
model_ivs1=scipy.io.loadmat('model_ivs1.mat')
M=model_ivs1.get('model_ivs1')
M=M*10;

# Get the Channel decomposition matrix
Q = V@(V.transpose())


# Retreive keys from storage
context = ts.context_from(read_data("data/public.txt"))

# Server Template Preparation
print('Server Preparation')
for c in range (0,151):
    print(c)
    # Encrypt template
    template =ts.ckks_vector(context, M[:,c])
    # Get template x Q <- Q in plain domain
    tempQ=template.matmul(Q)

    d1=tempQ.dot(template)

    # template_proto = template.serialize()
    tempQ_proto = tempQ.serialize()
    d1_proto = d1.serialize()


    s=str(c)

    # write_data("data/template_"+s+"_.txt", template_proto)
    write_data("data/tempQ_"+s+"_.txt", tempQ_proto)
    write_data("data/d1_"+s+"_.txt",d1_proto)




# Client Template Preparation
print('Client Preparation')
a=000000090.0
b=000000800.00
x_0=0.5*(1/sqrt(a) + 1/sqrt(b))

for r in range (0,302):
    print(r)
    # Encrypt test iVector
    test1 =ts.ckks_vector(context, T[:,r]*1.5*x_0)
    test2 =ts.ckks_vector(context, T[:,r]*(-0.5)*x_0*x_0*x_0)
    test3 =ts.ckks_vector(context, T[:,r])
    # Get template t Q <- Q in plain domain
    # test1Q=test1.matmul(Q)
    # test2Q=test2.matmul(Q)
    #
    test3p=test3.matmul(Q)
    test3Q=test3p.dot(test3)
    #
    #
    # test1Q_proto = test1Q.serialize()
    # test2Q_proto = test2Q.serialize()
    # test3Q_proto = test3Q.serialize()



    test1_proto = test1.serialize()
    test2_proto = test2.serialize()
    test3_proto = test3.serialize()
    test3Q_proto = test3Q.serialize()

    s=str(r)

    write_data("data/test1"+s+"_.txt", test1_proto)
    write_data("data/test2"+s+"_.txt", test2_proto)
    write_data("data/test3"+s+"_.txt", test3_proto)
    write_data("data/test3Q"+s+"_.txt", test3Q_proto)

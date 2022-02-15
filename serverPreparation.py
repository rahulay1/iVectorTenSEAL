import tenseal as ts
import base64
import scipy.io

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
context = ts.context_from(read_data("data81924134Scaledby10/public.txt"))

# Server Template Preparation

for c in range (0,151):
    # Encrypt template
    template =ts.ckks_vector(context, M[:,c])
    # Get template x Q <- Q in plain domain
    tempQ=template.matmul(Q)

    d1=tempQ.dot(template)

    template_proto = template.serialize()
    tempQ_proto = tempQ.serialize()
    d1_proto = d1.serialize()



    s=str(c)

    write_data("data81924134Scaledby10/template_"+s+"_.txt", template_proto)
    write_data("data81924134Scaledby10/tempQ_"+s+"_.txt", tempQ_proto)
    write_data("data81924134Scaledby10/d1_"+s+"_.txt",d1_proto)
    print("Server ",c)




# Client Template Preparation

for r in range (0,302):
    # Encrypt test iVector
    test =ts.ckks_vector(context, T[:,r])
    # Get template t Q <- Q in plain domain
    testQ=test.matmul(Q)
    d2=testQ.dot(test)

    test_proto = test.serialize()
    testQ_proto = testQ.serialize()
    d2_proto=d2.serialize()

    s=str(r)

    write_data("data81924134Scaledby10/test_"+s+"_.txt", test_proto)
    write_data("data81924134Scaledby10/testQ_"+s+"_.txt", testQ_proto)
    write_data("data81924134Scaledby10/d2_"+s+"_.txt", d2_proto)
    print("Client ",r)

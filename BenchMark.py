import tenseal as ts
import base64
import time
import numpy as np
import scipy.io
from math import sqrt
import psutil


start_time = time.time()


# context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree = 4096, coeff_mod_bit_sizes = [37, 17,17, 39])
# context.generate_galois_keys()
# context.global_scale = 2**17
#
# context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree = 8192, coeff_mod_bit_sizes = [41,26,26,26,26,26,41])
# context.generate_galois_keys()
# context.global_scale = 2**26

context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree = 16384, coeff_mod_bit_sizes = [60,35,35,35,35,35,35,35,35,35,60])
context.generate_galois_keys()
context.global_scale = 2**35

def get_cpu_frequency():
    """
    Obtains the real-time value of the current CPU frequency.
    :returns: Current CPU frequency in MHz.
    :rtype: int
    """
    return int(psutil.cpu_freq().current)

def get_ram_usage():
    """
    Obtains the absolute number of RAM bytes currently in use by the system.
    :returns: System RAM usage in bytes.
    :rtype: int
    """
    return int(psutil.virtual_memory().total - psutil.virtual_memory().available)

def get_cpu_usage_pct():
    """
    Obtains the system's average CPU load as measured over a period of 500 milliseconds.
    :returns: System CPU load as a percentage.
    :rtype: float
    """
    return psutil.cpu_percent(interval=0.5)


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


secret_context = context.serialize(save_secret_key = True)
write_data("data/secret.txt", secret_context)

#drop the secret_key from the context

# context.make_context_public()
public_context = context.serialize()
write_data("data/public.txt", public_context)

finishedKG=time.time()

print("Key Generation : %s seconds " % (finishedKG- start_time))





d=200;
template=np.random.rand(d)
test=np.random.rand(d)
Q=np.random.rand(d,d)

x_0=1/sqrt((test@Q@test)*(template@Q@template))

score_plain =(template@Q@test)*(x_0)



start_enc_vector = time.time()
# Encrypt template
templateE =ts.ckks_vector(context, template)
testE =ts.ckks_vector(context, test)
# Get template x Q <- Q in plain domain
end_enc_vector = time.time()


templateE_proto = templateE.serialize()
testE_proto = testE.serialize()

write_data("data/templateE.txt", templateE_proto)
write_data("data/testE.txt", testE_proto)



start_enc_vector_matmul = time.time()

tempQ=templateE.matmul(Q)

end_enc_vector_matmul = time.time()

print("Time to encrypt vector : %s seconds " % ((end_enc_vector- start_enc_vector)/2))
print("Time for enc vector mat mul : %s seconds " % (end_enc_vector_matmul- start_enc_vector_matmul))

# Server compute test x tQ

start_sever_verification = time.time()

topValue=testE.dot(tempQ)

d1=tempQ.dot(templateE)

d21=testE.matmul(Q)
d2=d21.dot(testE)

denominator=d1.mul(d2)

inv11=ts.ckks_vector(context, [1.5*x_0])
inv12= -0.5*(x_0**3)
inv13=denominator.mul(inv12)
inv1=inv13.add(inv11)

print('RAM usage is {} MB'.format(int(get_ram_usage() / 1024 / 1024)))
print('CPU frequency is {} MHz'.format(get_cpu_frequency()))
print('System CPU load is {} %'.format(get_cpu_usage_pct()))
inv21=inv1.mul(1.5)
inv22=(inv1.pow(3) ).mul(-0.5)
inv23=denominator.mul(inv22)
inv2=inv23.add(inv21)


score=topValue.mul(inv2)

end_sever_verification = time.time()
print("Time for server verification : %s seconds " % (end_sever_verification- start_sever_verification))


start_client_dec = time.time()
Enc_Score = score.decrypt()[0]
end_client_dec = time.time()

print("Time for user decryption : %s seconds " % (end_client_dec- start_client_dec))

print("Plain score :",score_plain)
print("Enc score :",Enc_Score)

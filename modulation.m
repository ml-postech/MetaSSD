function sym_vec = modulation(M)

M = cast(M, 'double');
load('codedBits.mat');

sym_vec = qammod(codedBits', M, 'InputType', 'Bit');
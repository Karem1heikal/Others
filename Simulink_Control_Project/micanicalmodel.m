%Transfer function to stadyspace

[a,b,c, d]= tf2ss ([1], [1 6.56 107])

% steadyspace to Transfer function
[num, den]=ss2tf ([ -6.56 -107;1 0], [1;0], [0 1], [0]) 
% transient Response obtain unit step Response

figure (1)
stp=tf (1, [1 6.56 107]);
step(stp)
grid

% transient Response obtain ramp Response

figure (2)

imp=tf(1, [1 6.56 107])

impulse (imp)

% response to arbitrary input

figure (3)

num=[1];

den=[1 6.56 107];

t=[0:.1:10];

inp=t.^0; %unit step
lsim (num, den, inp, t)

figure (4)

num=[1];
den=[1 6.56 107];
t=[0:.1:10];
inp=t; %ramp
lsim (num, den, inp, t);

%tf2zp Transfer function to zero-pole conve 
num=[1]; 
den=[1 6.56 107];
[z,p,k]=tf2zp (num, den);

%zp2tf zero-pole conversion to Transfer function.

z =[] ;
p=[-3.2800 + 9.8103i -3.2800 - 9.8103i];
k = 1 ;

[num, den]=zp2tf (z,p,k);

num=[1];
den=[1 6.56 107];

sys= tf(num, den);
P= pole(sys)
Z=zero(sys)
pzmap(sys)
% open loop tf
num=[1];
den=[1 6.56 105];
%root locus
figure (5)
rlocus (num, den)
%at sita =0.317 
sgrid([0.317],[])
figure (6)
nyquist (num, den)
figure (7)
margin (num, den)
figure (8)
bode (num, den)
figure (9)
nichols (num, den)
%block digram
g1=tf([1],[1 0]);
H1=4.08/0.622;
G1=feedback(g1,H1)
H2=66.6/0.622;
G2=series(G1,g1);
G3=feedback(G2,H2)


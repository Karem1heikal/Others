
s= tf('s');

m=0.622;
b1=4.08/3;
b2=4.08/3;
b3=4.08/3;
k1=66.6/2;
k2=66.6/2;

G=1/(m*s^2+(b1+b2+b3)*s+(k1+k2));
f=4;
y=G*f;

%electrical model RLC series
R=1;
L=1e-3;
C=1e-6;



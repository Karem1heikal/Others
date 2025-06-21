% BASK Modulation and Demodulation Simulation with BER Analysis
clear all;
close all;
clc;

% Simulation Parameters
N = 10000;          % Number of bits
Fs = 100;           % Sampling frequency
fc = 10;            % Carrier frequency
T = 1;              % Bit period
t = 0:1/Fs:T-1/Fs;  % Time vector for one bit
Eb = 1;             % Energy per bit

% Transmitter
% Generate random binary stream
data = randi([0 1], 1, N);

% Baseband Signal (Amplitude Modulation)
A1 = sqrt(2*Eb);   % Amplitude for '1'
A0 = 0;            % Amplitude for '0'

baseband_signal = data * A1; % Binary data scaled to amplitude

% Generate Carrier Signal
carrier = cos(2*pi*fc*t);

% Modulation
modulated_signal = [];
for i = 1:N
    modulated_bit = baseband_signal(i) * carrier;
    modulated_signal = [modulated_signal modulated_bit];
end

% Time vector for the entire modulated signal
t_total = 0:1/Fs:(N*T-1/Fs);

% Plot Baseband and Passband Signals
figure(1);
subplot(2,1,1);
stem(data(1:20), 'LineWidth', 2);
title('Original Binary Data (First 20 bits)');
ylabel('Amplitude');
grid on;

subplot(2,1,2);
plot(t_total(1:200), modulated_signal(1:200), 'LineWidth', 2);
title('BASK Modulated Signal (First 2 bits)');
xlabel('Time');
ylabel('Amplitude');
grid on;

% Spectrum Analysis
figure(2);
[pxx, f] = pwelch(modulated_signal, [], [], [], Fs);
plot(f, 10*log10(pxx), 'LineWidth', 2);
title('Power Spectral Density of BASK Signal');
xlabel('Frequency (Hz)');
ylabel('Power/Frequency (dB/Hz)');
grid on;

% Channel
EbN0dB = 0:2:20; % Eb/N0 range in dB
EbN0 = 10.^(EbN0dB/10); % Convert to linear scale
BER_simulated = zeros(size(EbN0));
BER_theoretical = qfunc(sqrt(EbN0/2)); % Theoretical BER for BASK

for i = 1:length(EbN0)
    % Noise Variance
    noise_variance = 1 / (2*EbN0(i)); % Correct noise variance
    noise = sqrt(noise_variance) * randn(size(modulated_signal));

    % Add Noise to Modulated Signal
    received_signal = modulated_signal + noise;

    % Receiver
    demodulated_signal = [];
    for j = 1:N
        % Extract the bit segment
        bit_segment = received_signal((j-1)*length(t)+1:j*length(t));
        % Correlation with carrier (coherent detection)
        decision_metric = sum(bit_segment .* carrier);
        % Threshold decision: compare against 0
        demodulated_bit = decision_metric > (sum(carrier.^2) * A0 + A1) / 2; 
        demodulated_signal = [demodulated_signal demodulated_bit];
    end

    % Calculate BER
    errors = sum(data ~= demodulated_signal);
    BER_simulated(i) = errors / N;
end

% Plot BER Curves
figure(3);
semilogy(EbN0dB, BER_theoretical, 'b-', 'LineWidth', 2);
hold on;
semilogy(EbN0dB, BER_simulated, 'r*-', 'LineWidth', 2);
grid on;
title('BER Performance of BASK');
xlabel('Eb/N0 (dB)');
ylabel('Bit Error Rate');
legend('Theoretical', 'Simulated');
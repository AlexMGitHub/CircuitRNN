#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 18:42:27 2020

@author: alex

###############################################################################
# Circuitry.py
#
# Revision:     1.00
# Date:         8/1/2020
# Author:       Alex
#
# Purpose:      Python implementations of digital and analog circuits used to
#               generate time series data.
#
# Classes:
# 1. NAND           -- A NAND gate
# 2. DLatch         -- A D-Latch composed of NAND gates
# 3. DFlipFlp       -- A D Flip-Flop composed of D-Latches and a NAND gate
# 4. ShiftRegister  -- A shift register composed of D Flip-Flops
# 5. DAC            -- A Digital-to-Analog Converter
# 6. RC_Circuit     -- A series resistor and capacitor low-pass filter
# 7. Comparator     -- An analog comparator that returns HIGH or LOW
#
# Functions:
# 1. test_RC        -- A test function that verifies the RC_Circuit() class
#
##############################################################################
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal


class NAND:
    """A class representing a NAND logic gate."""
    def __init__(self):
        """Initialize inputs A and B to zero"""
        self.A = 0
        self.B = 0
            
    def set_inputs(self, inputA, inputB):
        """Set the inputs of the NAND gate and return the resulting output."""
        self.A = inputA
        self.B = inputB
        if self.A and self.B:
            self.output = 0
        else:
            self.output = 1
        return self.output


class DLatch:
    """A class representing a D-Latch circuit.  The D-Latch is 
    composed of NAND gates implemented using the NAND() class.
    """
    def __init__(self):
        """Instantiate the NAND gates and initialize the latch outputs."""
        self.nand0 = NAND()
        self.nand1 = NAND()
        self.nand2 = NAND()
        self.nand3 = NAND()
        self.nand4 = NAND()
        self.Q = 0
        self.Q_bar = 1

    def clock_inputs(self, inputD, clock=1):
        """Accept an input from the D input pin of the latch.  Propagate
        input through the NAND gates and return the outputs of the latch.
        Assume PRESET and CLEAR pins are always HIGH.  Clock is applied to 
        enable pin of D-latch.
        """
        self.D = inputD
        self.clock = clock
        invert = self.nand0.set_inputs(inputD, inputD)
        out1 = self.nand1.set_inputs(inputD, self.clock)
        out2 = self.nand2.set_inputs(self.clock, invert)
        self.Q = self.nand3.set_inputs(out1, self.Q_bar)
        self.Q_bar = self.nand4.set_inputs(out2, self.Q)
        # Update Q again to ensure it's settled to its final value
        self.Q = self.nand3.set_inputs(out1, self.Q_bar)
        return self.Q, self.Q_bar        


class DFlipFlop:
    """A class representing a D Flip Flop circuit.  The D Flip-Flop is 
    composed of D-Latch circuits implemented using the DLatch() class.
    """
    def __init__(self):
        """Instantiate the D-Latch circuits and the NAND gate inverter.
        Initialize the flip-flop outputs.
        """
        self.nand0 = NAND()
        self.dlatch0 = DLatch()
        self.dlatch1 = DLatch()
        self.Q = 0
        self.Q_bar = 1

    def clock_inputs(self, inputD, clock=1):
        """Accept an input from the D input pin of the flip flop.  Propagate
        input through the D-Latches and return the outputs of the flip-flop.
        Assume PRESET and CLEAR pins are always HIGH.  Clock = 1 is a full
        clock cycle that goes from falling edge to rising edge.  Clock = 0 is 
        a full cycle from rising edge to falling edge.
        """
        self.D = inputD
        self.clock = clock
        self.clock = abs(self.clock - 1) # Flip clock to opposite level
        invert = self.nand0.set_inputs(self.clock, self.clock)
        Q0, Q0_bar = self.dlatch0.clock_inputs(inputD, invert)
        self.Q, self.Q_bar = self.dlatch1.clock_inputs(Q0, self.clock)
        self.clock = abs(self.clock - 1) # Flip clock back to original level
        invert = self.nand0.set_inputs(self.clock, self.clock)
        Q0, Q0_bar = self.dlatch0.clock_inputs(inputD, invert)
        self.Q, self.Q_bar = self.dlatch1.clock_inputs(Q0, self.clock)
        return self.Q, self.Q_bar


class ShiftRegister:
    """A class representing an N-bit Single Input Parallel Output (SIPO) 
    shift register.  The shift register is composed of D-Latches implemented
    using the DLatch() class.
    """
    def __init__(self, nbits):
        """Initialize shift register with desired number of bits, and 
        instantiate N-bits number of DFlipFlop() instances.
        """
        self.nbits = nbits
        self.dff = [DFlipFlop() for i in range(nbits)]
        self.Q = [0 for x in range(0, nbits)]
        
    def clock_inputs(self, data):
        """Clock new input into first flip-flop, and shift all previous
        bits down one flip-flop.  Return list containing values stored in each 
        flip-flop.
        """
        for idx in range(self.nbits-1, 0, -1):
            self.dff[idx].clock_inputs(self.dff[idx-1].Q)
        self.dff[0].clock_inputs(data)
        self.Q = [dff.Q for dff in self.dff]
        return self.Q
    
        
class DAC:
    """A class representing an N-bit Digital-to-Analog Converter (DAC)."""
    def __init__(self, nbits, vref=1):
        """Initialize DAC with the desired number of bits and a symmetric 
        reference voltage for the analog output (+/- VRef).
        """
        self.nbits = nbits # Number of bits of DAC
        self.dbits = np.zeros(nbits, dtype=np.uint8) # Data bits
        self.vref = vref # Assume symmetric +/- Vref voltage
        # Derived parameters
        self.num_levels = 2 ** nbits # Number of output voltage levels
        
    def clock_inputs(self, data_bits):
        """Accept a list of bits and convert them to an analog voltage using
        offset binary.  Return the analog voltage as a float.
        
        Digital Input   | Analog Output
        MSB     LSB
        1111 1111       = +VRef*(127/128)
        1000 0001       = +VRef*(1/128)
        1000 0000       = 0
        0111 1111       = -VRef*(1/128)
        0000 0001       = -VRef*(127/128)
        0000 0000       = -VRef*(128/128)
        """
        # Index zero corresponds to D0 (LSB), index -1 corresponds to the n-1
        # data bit (MSB). E.g. 10010000 = 9 in unsigned binary to decimal
        assert len(data_bits) == self.nbits, 'Expect {}-bit input'.format(self.nbits)
        for idx in range(self.nbits):
            self.dbits[idx] = data_bits[idx]    
        # Convert digital input to voltage output assuming bipolar 
        # offset binary code table
        decimal = int(np.packbits(self.dbits, bitorder='little'))
        offset_decimal = decimal - (self.num_levels / 2)
        self.output_voltage = self.vref * offset_decimal / (self.num_levels / 2)
        return self.output_voltage


class RC_Circuit:
    """A class representing the transient response of a voltage applied to a 
    series resistor and capacitor.
    """
    def __init__(self, tau, vinit=0):
        """Initialize RC circuit with desired time constant and  
        steady state voltage.
        """
        self.vinit = vinit # Capacitor's initial voltage
        self.vapplied = vinit # Voltage applied to RC circuit
        self.vcap = vinit # Capacitor's current voltage
        self.t = 0 # Duration of time voltage has been applied to RC circuit
        self.tau = tau # RC time constant
       
    def apply_voltage(self, vapplied, tdelta):
        """Apply a voltage to the RC circuit for a specified period of time.
        Calculate the transient response of the RC circuit and return the 
        voltage across the capacitor.
        """
        if vapplied == self.vapplied: # Applied voltage has not changed
            self.t += tdelta 
        else: # Applied voltage has changed
            self.t = tdelta 
            self.vinit = self.vcap
            self.vapplied = vapplied
        self.vcap = vapplied - (vapplied - self.vinit)*np.exp(-self.t/self.tau)
        return self.vcap


class Comparator:
    """A class representing a simple analog comparator circuit."""
    def __init__(self, vthreshold, vsupply_pos=1, vsupply_neg=0):
        """Initialize comparator with voltage threshold and supply rails."""
        self.vthreshold = vthreshold
        self.vpos = vsupply_pos
        self.vneg = vsupply_neg
        self.vout = vsupply_neg
        
    def apply_voltage(self, vapplied):
        """Compare input voltage to threshold and return output voltage."""
        if vapplied > self.vthreshold:
            self.vout = self.vsupply_pos
        else:
            self.vout = self.vsupply_neg
        

def test_RC(tau=1, period=10, duration=20, offset=0, amplitude=1):
    """Test function to verify RC circuit output matches expectations."""
    ncycles = int(duration / period)
    duration = ncycles * period # Enforce integer number of cycles
    t = np.linspace(0, duration, 100*ncycles, endpoint=False)
    swave = amplitude * signal.square(2 * np.pi * (1/period) * t) + offset
    voltage_list = []
    test_circuit = RC_Circuit(tau, offset-amplitude)
    voltage_list.append(test_circuit.vinit)
    for i in range(1,len(t)):
        voltage = test_circuit.apply_voltage(swave[i], period/100)
        voltage_list.append(voltage)
    plt.figure(figsize=(12,9))    
    plt.plot(t, voltage_list, label='Vcap')
    plt.plot(t, swave, label='Input Voltage')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    test_RC()

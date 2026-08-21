# Apartment-suitable scanner prototype

The acceptance fixture is based on a compact Pololu Romi platform so the first
physical prototype can be assembled at a desk and operated at low voltage.
Prices are informational snapshots from 2026-08-06 and exclude tax/shipping.

## Core parts

* Romi Chassis Kit, item 3500: two 120:1 HP motors, 70 mm wheels, caster,
  six-AA holder, and chassis (`$39.95`).
* Romi 32U4 Control Board, item 3544: dual motor drivers, encoder inputs,
  accelerometer/gyro, MCU, and Raspberry Pi interface (`$74.95`).
* Romi Encoder Pair Kit, item 3542 (`$9.95`).
* Robot Arm Kit for Romi, item 3550: lift/tilt mechanisms and position-feedback
  servos (`$99.95`). The gripper is replaced with a light camera cradle.
* A second Romi ball caster (`about $3.95`) for arm stability.
* Raspberry Pi Zero 2 W (`from $15`), microSD card, Zero camera cable, and
  Raspberry Pi Camera Module 3 Wide (`from $35`; standard starts at `$25`).
* 6 V S13V25F6 step-up/step-down regulator (`$18.95`) for the arm servos. Its
  2.5 A typical capacity requires slew-limited arm moves; characterize the real
  load before allowing simultaneous fast reversals.
* Six matched low-self-discharge AA NiMH cells, a reputable matched-cell charger,
  small wire, connectors, heat-shrink, strain relief, a fuse selected after
  measured current, and an accessible latching power switch.

The known core listed-price subtotal is roughly `$299` before cells, storage,
cables, mounting consumables, shipping, and tools. Verify price and stock before
ordering.

## Tools

Required or strongly recommended: temperature-controlled soldering iron with a
stable stand, local fume extraction or effective ventilation, safety glasses,
flush cutters, wire stripper, small Phillips drivers, needle-nose pliers,
heat-shrink and heat source, and a digital multimeter. A small crimp tool is
useful if you standardize on crimp housings.

A 3D printer is **not required**. The Romi arm/chassis are injection-molded; the
camera cradle can be ordered from a print service, cut from thin sheet, or made
from a commercial camera bracket. If you buy a printer for repeated iterations,
prefer an enclosed FDM unit with source extraction/filtration, use lower-emission
material where practical, and place it away from occupied living/sleeping space.
NIOSH recommends substitution and engineering controls such as ventilation and
HEPA filtration for printer emissions.

## Assembly adaptation

Assemble the manufacturer kits first. Omit the gripper at the final arm joint
and attach a camera cradle whose optical axis points 90 degrees left of the
robot's forward axis. Keep the Pi on the chassis and route only the flexible
camera cable along the arm with a service loop and strain relief. For a
counter-clockwise orbit the chassis faces tangent to the circle and the left-
facing camera points inward. The lift servo creates the vertical sweep; the tilt
servo maintains vertical aim at the known object center.

Power the two arm servos from the dedicated regulated 6 V rail, join grounds at
the intended distribution point, and never power them from a GPIO pin. Ramp
commands, enforce mechanical-angle limits, monitor the feedback wires, and stop
on sustained position error or brownout. Perform the first powered tests with
the wheels raised, then on an open floor at low duty cycle.


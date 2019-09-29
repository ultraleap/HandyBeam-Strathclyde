"""

module :mod:`strathclyde` -- complementary material for the RCNDE Ultrasonic Transduction Course

Contains the class :class:`LinearArray`.


"""
from os import linesep
import numpy as np
import matplotlib.pyplot as plt
from handybeam.misc import HandyDict



tau = 2 * np.pi
"""Tau is the *actual* circle constant. 

Not some half-circle constant like pi.
"""


class LinearArray:
    """
    Calculates some basic properties of a linear phased array probe, and then generates input data for HandyBeam and OnScale/PZFlex.
    """

    def __init__(
                 self,
                 radiation_frequency=40e3,
                 sound_velocity=343,
                 sampling_density=17,
                 passive_aperture=12e-3,
                 element_pitch=1.0e-3,
                 element_width=None,
                 element_count=16,
                 focal_point=np.array((0.0e-3, 0.0e-3, 200e-3)),
                 amplitude_setting=1.0,
                 window_coefficients=None
                ):
        """ Initialize the LinearArray object with the following properties:

        Example use:

        .. code-block:: python

            array_builder = strathclyde.LinearArray()  # initialize
            array_builder  # show output


        :param float radiation_frequency: intended fundamental radiation frequency
        :param float sound_velocity: sound velocity in the medium interfacing the probe
        :param int sampling_density: for output to HandyBeam, how many sampling points create per lambda.
        :param float passive_aperture: size of the passive aperture
        :param float element_pitch: distance between centres of elements
        :param float element_width: active-aperture size of the array element. Set to :code:`None` to get :code:`element_pitch/2`
        :param int element_count: How many elements in this array.
        :param tuple(float) focal_point: xyz location of the intended focal point.
        :param float amplitude_setting: amplitude setting of the elements; it's like a voltage setting. Use to beautify visualisation only, otherwise leave at 1.0
        :param np.array() window_coefficients: window coefficients for apodisation.
            Set to None to get rectangular window (boxcar, ones)
            Set to np.array(shape=(element_count,),dtype=np.float) to set particular coefficients.
            Example 1: :code:`scipy.signal.windows.flattop(16,sym=True)`
            Example 2: :code:`np.ones(shape=(16,))`
      """
        self.radiation_frequency = radiation_frequency
        self.sound_velocity = sound_velocity
        self.passive_aperture = passive_aperture
        self.sampling_density = sampling_density
        self.element_pitch = element_pitch
        self.element_count = element_count
        self.amplitude_setting = amplitude_setting
        if window_coefficients is None:
            self.window_coefficients = np.ones(shape=(self.element_count,))
        else:
            self.window_coefficients = window_coefficients
        if element_width is None:
            self.element_width = element_pitch * 0.5
        else:
            self.element_width = element_width

        self.focal_point = np.array(focal_point)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# @property*

    @property
    def wavelength(self):
        """ wave length in given media, in meters [m]

        does :code:`return self.sound_velocity/self.radiation_frequency`
        """
        return self.sound_velocity/self.radiation_frequency

    @property
    def wavenumber(self):
        """count of waves per meter, in [1/m]

        does :code:`return 1.0 / self.wavelength`
        """
        return 1.0 / self.wavelength

    @property
    def wavenumber_rotations(self):
        """count of rotations per meter, in [radians/m]

        does :code:`return 2.0*np.pi / self.wavelength`
        """
        return 2.0*np.pi / self.wavelength

    @property
    def dx_simulation(self):
        """distance between points for sampling the radiating surfaces with radiating points, in [m]

        does :code:`return self.wavelength / self.sampling_density`
        """
        return self.wavelength / self.sampling_density

    @property
    def active_aperture(self):
        """calculated size of the active aperture [m]

        does :code:`return self.element_pitch * self.element_count`
        """
        return self.element_pitch * self.element_count

    @property
    def element_gap(self):
        """calculated gap width between elements

        does :code:`return self.element_pitch-self.element_width`
        """
        return self.element_pitch-self.element_width

    @property
    def active_aperture_near_field(self):
        """estimated distance from the probe surface to the transition between near and far field,[m]

        does :code:`return self.active_aperture**2 * self.radiation_frequency / 4.0 / self.sound_velocity`
        """
        return self.active_aperture**2 * self.radiation_frequency / 4.0 / self.sound_velocity

    @property
    def passive_aperture_near_field(self):
        """estimated distance from the probe surface to the transition between near and far field,[m]

        does :code:`return self.passive_aperture**2 * self.radiation_frequency / 4.0 / self.sound_velocity`
        """
        return self.passive_aperture**2 * self.radiation_frequency / 4.0 / self.sound_velocity

    def focusing_power_estimate(self, focal_distance, aperture):
        """estimated size of the focal point,  given aperture, distance and environment parameters,[m]

        does :code:`return 1.02*focal_distance*self.sound_velocity/self.radiation_frequency/aperture`

        :param float focal_distance: distance from the centre of the transmitter array to the focal point
        :param float aperture: aperture (passive or active) of interest. The result is valid for the aperture given.
        :return: estimated size of the focal point, [meters]
        """
        return 1.02*focal_distance*self.sound_velocity/self.radiation_frequency/aperture

    @property
    def focal_distance(self):
        """calculated distance from probe centre to the selected focal point,[m]

        does :code:`return np.sqrt(np.sum(np.array(self.focal_point)**2))`
        """
        return np.sqrt(np.sum(np.array(self.focal_point)**2))

    @property
    def passive_aperture_focus_power_estimate(self):
        """estimated natural focus spot size in the passive plane

        does :code:`return self.focusing_power_estimate(self.passive_aperture_near_field, self.passive_aperture)`
        """
        return self.focusing_power_estimate(self.passive_aperture_near_field, self.passive_aperture)

    @property
    def active_aperture_focus_power_estimate(self):
        """estimated focal spot size in the active plane, for given selected focal point location

        does :code:`return self.focusing_power_estimate(self.focal_distance, self.active_aperture)`
        """
        return self.focusing_power_estimate(self.focal_distance, self.active_aperture)

    @property
    def element_centre_locations(self):
        """location of the centres of the elements

        format: :code:`np.array(n,3)`,

        where first dimension, :code:`n` - count of elements,

        2nd dimension: :code:`[idx,:]=np.array((x,y,z))` - `x`,`y`,`z` components of the location vector; `idx` - element index

        """
        x = np.zeros((1, self.element_count))
        z = np.zeros((1, self.element_count))
        y = np.expand_dims(np.linspace(start=-self.active_aperture / 2,
                                       stop=self.active_aperture / 2,
                                       num=self.element_count,
                                       endpoint=True
                                       ), axis=0)
        locations = np.concatenate((x, y, z)).T
        return locations

    def time_of_flight(self,
                       source=np.array((0.0, 0.0, 0.0)),
                       destination=np.array((0.0, 0.0, 1.0))
                       ):
        """calculate time of flight between two points

        Point is a tuple e.g. :code:`np.array(0.0,0.0,1.0)`

        does :code:`np.sqrt(np.sum((destination-source)**2))/self.sound_velocity`

        """
        return np.sqrt(np.sum((destination-source)**2))/self.sound_velocity

    @property
    def time_of_flight_probe_to_focal_point(self):
        """ calculate and return all Time of Flights between probe elements and the focal point

        :returns: tofs=np.array(), shaped [ self.element_count,]
        """

        locations = self.element_centre_locations
        tofs = np.full((self.element_count, 1), np.NaN)  # initialize the output with NaNs
        for idx in range(self.element_count):
            tofs[idx, 0] = self.time_of_flight(self.focal_point, locations[idx, :])

        return tofs

    @property
    def focal_laws_delays(self):
        """generate focal laws -- the **delays** needed for each array element to focus the wave into the focal point

        Example:

        .. code-block:: python

            array_builder=strathclyde.LinearArray()
            print(array_builder.focal_laws_delays)

            [0.00000000e+00 7.75228203e-06 1.49642195e-05 2.15665444e-05
             2.74851133e-05 3.26429389e-05 3.69632725e-05 4.03737335e-05
             4.28112228e-05 4.42270507e-05 4.45914479e-05 4.38965672e-05
             4.21573062e-05 3.94097568e-05 3.57076435e-05 3.11175211e-05]

        :returns: focal_laws_delays=np.array(), shaped [ self.element_count,]
        """
        tofs = self.time_of_flight_probe_to_focal_point
        return np.max(tofs)-tofs

    @property
    def radiation_period(self):
        """converts self.radiation_frequency to period"""
        return 1.0/self.radiation_frequency

    @property
    def focal_laws_phases(self):
        """converts time-domain focal_laws_delays into frequency-domain phases

        the phases are wrapped (that is, range from 0.0 to tau)

        does :code:`return np.mod(self.focal_laws_delays/self.radiation_period, 1.0)*tau`

        :returns: focal_laws_phases=np.array(), shaped [ self.element_count,]
        """

        return np.mod(self.focal_laws_delays/self.radiation_period, 1.0)*tau

    @property
    def focal_laws_gains(self):
        """ generates gains for elements -- e.g. includes apodisation or shadowing

        does :code:`return self.window_coeffs*self.amplitude_setting`

        :return:
        """
        return self.window_coefficients*self.amplitude_setting

    def create_point_cloud_for_array_element(self,
                                             passive_aperture=np.NaN,
                                             active_aperture=np.NaN,
                                             element_position=np.array((0.0, 0.0, 0.0)),
                                             tx_amplitude=1.0,
                                             tx_phase=0.0):
        """ Creates a cloud of :code:`xyznnnddddap____` points that simulate a larger aperture of a single phased array element.

        for the description of what is `xyznnnddddap____`, see the `tx_array_descriptor_a` documentation of `HandyBeam`

        Makes sure that the total output (as set by directivity_amplitude_poly2_c0) is set to be equal to the desired area of the transducer
        that is, the larger the transducer, the larger the total output.

        .. ToDo::

            turn the above codewords into links to the actual HandyBeam documentation.

        note: the sampling is adjusted so that the sampler points are always touching the edge of the desired aperture.


        :param passive_aperture: element size along the passive aperture, the x-dimension
        :param active_aperture: element size along the active aperture, or the y-dimension
        :param element_position: xyz location of the element.
        :param float tx_amplitude: the amplitude to set to this transducer
        :param float tx_phase: the phase to set to this transducer
        :return: numpy.array.shape == (:,16); the count of points generated depends on :code:`self.dx_simulation`

        """
        count_of_points_along_x = np.int(np.ceil(passive_aperture / self.dx_simulation))
        count_of_points_along_y = np.int(np.ceil(active_aperture / self.dx_simulation))
        coordinate_of_points_along_x = np.linspace(
            -passive_aperture / 2 + element_position[0],
            passive_aperture / 2 + element_position[0],
            count_of_points_along_x)

        coordinate_of_points_along_y = np.linspace(
            -active_aperture / 2 + element_position[1],
            active_aperture / 2 + element_position[1],
            count_of_points_along_y)

        element_area = passive_aperture * active_aperture
        count_of_sampling_points = count_of_points_along_x*count_of_points_along_y
        amplitude_scaling = element_area / count_of_sampling_points

        point_x_grid, point_y_grid = np.meshgrid(coordinate_of_points_along_x, coordinate_of_points_along_y)
        point_x_list = np.expand_dims(np.ravel(point_x_grid), axis=0).T
        point_y_list = np.expand_dims(np.ravel(point_y_grid), axis=0).T
        point_z_list = np.zeros(shape=point_x_list.shape)

        point_zeros_list = np.zeros(shape=point_x_list.shape)
        point_ones_list = np.ones(shape=point_x_list.shape)
        point_nan_list = np.full(point_x_list.shape, np.NaN)

        point_list = np.concatenate(
            (point_x_list,  # x
             point_y_list,  # y
             point_z_list,  # z
             point_zeros_list,  # xnormal
             point_zeros_list,  # ynormal
             point_ones_list,  # znormal
             point_zeros_list,  # directivity_phase_poly1_c1
             point_ones_list * amplitude_scaling,  # directivity_amplitude_poly2_c0 - scaled to produce output proportional to the area of the element
             point_zeros_list,  # directivity_amplitude_poly2_c1
             point_zeros_list,  # directivity_amplitude_poly2_c2
             point_zeros_list + tx_amplitude,  # amplitude_ratio_setting
             point_zeros_list + tx_phase,  # phase_setting
             point_nan_list,  # 1st nan
             point_nan_list,  # 2nd nan
             point_nan_list,  # 3rd nan
             point_nan_list   # last nan
             ), axis=1).astype(np.float32)

        return point_list

    def create_point_cloud_all_elements(self):
        """Creates the cloud of points for HandyBeam core propagator to consume.

        The multiple points correctly account for the directivity of an element with arbitrary aperture.

        The Handybeam's directivity model is not used. (coefficients at zero).

        :return: point cloud, numpy.array.shape == (:,16), the count of points generated depends on :code:`self.dx_simulation`


        """
        element_locations = self.element_centre_locations
        focal_laws = self.focal_laws_delays
        # convert time-domain focal law to frequency-domain focal law at the frequency of interest
        # this means, wrap
        focal_laws_phases = self.focal_laws_phases
        focal_laws_gains = self.focal_laws_gains

        point_list = np.empty((0, 16))
        for idx_element in range(self.element_count):
            point_list = np.concatenate(
                (point_list,
                 self.create_point_cloud_for_array_element(
                    passive_aperture=self.passive_aperture,
                    active_aperture=self.element_width,
                    element_position=element_locations[idx_element, :],
                    tx_amplitude=focal_laws_gains[idx_element],
                    tx_phase=focal_laws_phases[idx_element])
                 )
                 )
        return point_list.astype(np.float32)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# visualize*

    def __str__(self):
        """Returns basic properties of the probe definition

        Example output :

        .. code-block:: XML

            Basic linear probe:
            > Environment:
            >>   radiation frequency: 40.0kHz
            >>   sound_velocity :343.0m/s
            >>   sound wave length :8.575mm
            >>   medium wavenumber: 116.6[waves/meter]
            >>   point sources sampling density: 17pt/lambda linear, spacing of 0.504mm

            > Probe definition:
            >>   Passive aperture: 64.0mm
            >>   element width: 2.000mm
            >>   element count: 16

            > Probe calculated properties:
            >>   inter-element gap: 2.0mm
            >>   Active aperture: 64.0mm
            >>   Active aperture near field transition: 119.4mm
            >>   Passive aperture near field transition: 119.4mm

            > Focal point calculated properties:
            >>   focal distance: 51.0mm
            >>   active aperture -6dB focal spot size: 7.0mm
            >>   passive aperture -6dB natural focus spot size: 16.3mm

        """
        txt = ""
        txt = txt + "Basic linear probe:"
        txt = txt + linesep + "> Environment:"
        txt = txt + linesep + ">>   radiation frequency: {:0.1f}kHz".format(self.radiation_frequency*1e-3)
        txt = txt + linesep + ">>   sound_velocity :{:0.1f}m/s".format(self.sound_velocity)
        txt = txt + linesep + ">>   sound wave length :{:0.3f}mm".format(self.wavelength*1e3)
        txt = txt + linesep + ">>   medium wavenumber: {:0.1f}[waves/meter]".format(self.wavenumber)
        txt = txt + linesep + ">>   point sources sampling density: {}pt/lambda linear, spacing of {:0.3f}mm".format(self.sampling_density, self.dx_simulation*1e3)
        txt = txt + linesep + "  "
        txt = txt + linesep + "> Probe definition:"
        txt = txt + linesep + ">>   Passive aperture: {:0.1f}mm".format(self.passive_aperture*1e3)
        txt = txt + linesep + ">>   element width: {:0.3f}mm".format(self.element_width*1e3)
        txt = txt + linesep + ">>   element count: {}".format(self.element_count)
        txt = txt + linesep + "  "
        txt = txt + linesep + "> Probe calculated properties:"
        txt = txt + linesep + ">>   inter-element gap: {:0.1f}mm".format(self.element_gap*1e3)
        txt = txt + linesep + ">>   Active aperture: {:0.1f}mm".format(self.active_aperture*1e3)
        txt = txt + linesep + ">>   Active aperture near field transition/boundary: {:0.1f}mm".format(self.active_aperture_near_field*1e3)
        txt = txt + linesep + ">>   Passive aperture near field transition/boundary: {:0.1f}mm".format(self.passive_aperture_near_field * 1e3)
        txt = txt + linesep + "  "
        txt = txt + linesep + "> Focal point calculated properties:"
        txt = txt + linesep + ">>   focal distance: {:0.1f}mm".format(self.focal_distance*1e3)
        txt = txt + linesep + ">>   active aperture -6dB focal spot size: {:0.1f}mm".format(self.active_aperture_focus_power_estimate * 1e3)
        txt = txt + linesep + ">>   passive aperture -6dB natural focus spot size: {:0.1f}mm".format(self.passive_aperture_focus_power_estimate * 1e3)
        return txt

    def __repr__(self):
        """links to self.__str__()"""
        return self.__str__()

    @property
    def stats(self):
        """ return calculated properties in a dictionary format (HandyDict)

        HandyDict is a dict, but with some extra methods so that a dot notation works on it.

        :return: HandyDict stats -  a dictionary with stats saved
        """
        stats = HandyDict({
            'radiation_frequency': self.radiation_frequency,
            'sound_velocity': self.sound_velocity,
            'wavelength': self.wavelength,
            'wavenumber': self.wavenumber,
            'sampling_density_setting': self.sampling_density,
            'sampling_density_effective': self.dx_simulation,
            'passive_aperture': self.passive_aperture,
            'element_width': self.element_width,
            'element_count': self.element_count,
            'element_gap': self.element_gap,
            'active_aperture': self.active_aperture,
            'active_aperture_nearfield_boundary': self.active_aperture_near_field,
            'passive_aperture_nearfield_boundary': self.passive_aperture_near_field,
            'current_focal_point_distance': self.focal_distance,
            'active_aperture_focus_power_estimate': self.active_aperture_focus_power_estimate,
            'passive_aperture_focus_power_estimate': self.passive_aperture_focus_power_estimate
            }
            )

        return stats

    def visualize_array_elements(self, figsize=(4, 3), dpi=150, filename=None):
        """ create a 2D top-down plot of how do the array element patches look like

        Example output :

        .. image:: _static/example_visualize_array_elements.png

        :param figsize: size in inches of the created figure canvas
        :param dpi: resolution in points-per-inch for the figure canvas
        :param filename: file name to save to. For example, :code:`output.png`. Do attach a `.png` at the end. Default :code:`None`
        :return: displays a figure, and optionally, saves it to a file.
        """

        """create a 2D top-down plot of how do the array element patches look like"""
        hf = plt.figure(figsize=figsize, dpi=dpi)
        ha = plt.axes()

        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Rectangle

        elements = []
        locations = self.element_centre_locations
        for idx in range(self.element_count):
            element = Rectangle((-self.passive_aperture / 2, locations[idx, 1] - self.element_width / 2),
                                self.passive_aperture, self.element_width)
            elements.append(element)
        pc = PatchCollection(elements, facecolor='r', alpha=0.5,
                             edgecolor=None)
        ha.add_collection(pc)

        # add points for element centres
        # hp = plt.plot(self.element_centre_locations[:, 0], self.element_centre_locations[:, 1], 'o')

        plt.grid(True)
        plt.axis('equal')
        plt.xlabel('x-dimension[m]\npassive aperture')
        plt.ylabel('y-dimension[m]\nactive aperture')

        if filename is None:
            plt.show()
        else:
            plt.savefig(filename, bbox_inches='tight')
            plt.close()

    def visualize_time_of_flight(self, figsize=(4, 3), dpi=150, filename=None):
        """plots the computed time-of-flight values for each probe element

        Example output:

        .. image:: _static/example_visualize_time_of_flight.png

        :param figsize: size in inches of the created figure canvas
        :param dpi: resolution in points-per-inch for the figure canvas
        :param filename: file name to save to. For example, :code:`output.png`. Do attach a `.png` at the end. Default :code:`None`
        :return: displays a figure, and optionally, saves it to a file.

        """
        plt.figure(figsize=figsize, dpi=dpi)
        plt.stem(np.arange(0, self.element_count), self.time_of_flight_probe_to_focal_point * 1e6)
        plt.xticks(np.arange(0, self.element_count, step=4))
        plt.xlabel('element index[-]')
        plt.ylabel('time of flight from probe\nto focal point[$\mu$s]')
        plt.grid(True)
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename, bbox_inches='tight')
            plt.close()

    def visualize_focal_laws(self, figsize=(4, 3), dpi=150, filename=None):
        """plots computed firing delays (focal laws)  for each probe element

        Example output :

        .. image:: _static/example_visualize_focal_laws.png


        :param figsize: size in inches of the created figure canvas
        :param dpi: resolution in points-per-inch for the figure canvas
        :param filename: file name to save to. For example, :code:`output.png`. Do attach a `.png` at the end. Default :code:`None`
        :return: displays a figure, and optionally, saves it to a file.


        """
        plt.figure(figsize=figsize, dpi=dpi)
        plt.stem(np.arange(0, self.element_count), self.focal_laws_delays * 1e6)
        plt.xlabel('element index[-]')
        plt.ylabel("time of flight from probe\nto focal point[$\mu$s]")
        plt.xticks(np.arange(0, self.element_count, step=4))
        plt.grid(True)
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename, bbox_inches='tight')
            plt.close()

    def visualize_point_cloud_all_elements(self, figsize=(4, 3), dpi=150, filename=None):
        """plots computed firing delays (focal laws)  for each probe element

        Example output :

        .. image:: _static/example_point_cloud_all_elements.png

        :param figsize: size in inches of the created figure canvas
        :param dpi: resolution in points-per-inch for the figure canvas
        :param filename: file name to save to. For example, :code:`output.png`. Do attach a `.png` at the end. Default :code:`None`
        :return: displays a figure, and optionally, saves it to a file.
        """

        plt.figure(figsize=figsize, dpi=dpi)
        point_cloud = self.create_point_cloud_all_elements()
        plt.plot(point_cloud[:, 0], point_cloud[:, 1], ',')
        plt.grid(True)
        plt.axis('equal')
        plt.xlabel('x-dimension[m]\npassive aperture')
        plt.ylabel('y-dimension[m]\nactive aperture')
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename, bbox_inches='tight')
            plt.close()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Export*

    def export_focal_laws_to_onscale(self, filename="focal_laws_delays.txt"):
        """ produces an input file to PZFlex/Onscale, that contains the focal laws and per-element gains

        Example output file contents:

        .. code-block:: XML

            symb tshift1 = 0.000000e+00
            symb eweight1 = -4.210510e-04
            symb tshift2 = -4.218830e-07
            symb eweight2 = -6.061185e-03
            symb tshift3 = -8.132675e-07
            symb eweight3 = -3.136265e-02

        :param string filename: file name to use. Remember about adding the file extension. `.txt` is recommended.

        :return: prints the code to screen, and saves to the specified file.


        """
        delays = self.focal_laws_delays
        gains = self.focal_laws_gains
        gains = np.expand_dims(gains,1)
        txt = ""+linesep
        for idx in range(self.element_count):
            txt_delay = "symb tshift{} = {:0.6e}".format(idx, delays[idx, 0])
            txt_gain = "symb eweight{} = {:0.6e}".format(idx, gains[idx, 0])
            txt = txt+txt_delay+linesep+txt_gain+linesep

        # save to file now

        handle_to_file = open(filename, "w+")
        handle_to_file .write(txt)
        handle_to_file .close()

        return txt

    def create_handybeam_world(self):
        """ creates a handybeam.world.World object with the settings set up so that it is ready to conduct the simulation of the acoustic field generated by the array.

        :return handybeam.world.World: the :class:`handybeam.world.World` object with :code:`tx_array` and :code:`samplers` set up

        """
        import handybeam
        import handybeam.world
        import handybeam.tx_array
        import handybeam.tx_array_library
        world = handybeam.world.World(frequency=self.radiation_frequency, sound_velocity=self.sound_velocity)
        hb_array = handybeam.tx_array.TxArray()
        hb_array.tx_array_element_descriptor = self.create_point_cloud_all_elements()
        hb_array.name = "Strathclyde style linear array, subsampled, {} real elements".format(self.element_count)
        world.tx_array = hb_array

        return world

# ==========================================================================================
# ==========================================================================================
# ==========================================================================================
# ==========================================================================================
# ==========================================================================================
# visualize* that need no object reference


def visualize_2d_amplitude(sampler,figsize=(4, 3), dpi=150, filename=None):
    """this visualizes using sampler.extent -- that is, coordinates relative to the grid"""
    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(np.abs(sampler.pressure_field), cmap='hot', extent=sampler.extent)
    plt.xlabel('x-extent of the grid[m]')
    plt.ylabel('y-extent of the grid[m]')
    lims = np.max(np.abs(np.nan_to_num(sampler.pressure_field)))
    plt.clim(0, lims * 0.8)
    plt.colorbar()
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, bbox_inches='tight')
        plt.close()


def get_rectilinear_sampler_coordinates_maxmin(coordinates):
    x = coordinates[:, :, 0].ravel()
    xmax = np.max(x)
    xmin = np.min(x)
    y = coordinates[:, :, 1].ravel()
    ymax = np.max(y)
    ymin = np.min(y)
    z = coordinates[:, :, 2].ravel()
    zmax = np.max(z)
    zmin = np.min(z)
    return (xmax, xmin, ymax, ymin, zmax, zmin)


def visualize_2d_amplitude_xz(sampler, figsize=(4, 3), dpi=150, filename=None,xlabel="provide x-label name",ylabel="provide y-label name"):
    """ this visualizes using sampler.coordinates -- that is, coordinates are absolute to the world

    :param sampler: the sampler object
    :param figsize: figure size in inches
    :param dpi: figure resolution in dpi
    :param filename: if set, this is the file name to write. do not forget the :code:`.png` extension.
    :param string xlabel: set to desired text on the x-axis of the figure. This will depend on what sampler do You provide.
    :param string ylabel: set to desired text on the x-axis of the figure. This will depend on what sampler do You provide.
    :return: displays or saves the figure.
    """
    plt.figure(figsize=figsize, dpi=dpi)
    limits = get_rectilinear_sampler_coordinates_maxmin(sampler.coordinates)
    plt.imshow(np.abs(sampler.pressure_field), cmap='hot', extent=(limits[2], limits[3], limits[5], limits[4]))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    lims = np.max(np.abs(np.nan_to_num(sampler.pressure_field)))
    plt.clim(0, lims * 0.8)
    plt.colorbar()
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, bbox_inches='tight')
        plt.close()


def visualize_2d_real(sampler, figsize=(4, 3), dpi=150,
                      filename=None,
                      xlabel="provide x-label name",
                      ylabel="provide y-label name"):
    """ this visualizes using sampler.coordinates -- that is, coordinates are absolute to the world

    :param sampler: the sampler object
    :param figsize: figure size in inches
    :param dpi: figure resolution in dpi
    :param filename: if set, this is the file name to write. do not forget the :code:`.png` extension.
    :param string xlabel: set to desired text on the x-axis of the figure. This will depend on what sampler do You provide.
    :param string ylabel: set to desired text on the x-axis of the figure. This will depend on what sampler do You provide.
    :return: displays or saves the figure.
    """
    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(np.real(sampler.pressure_field), cmap='bwr', extent=sampler.extent)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    lims = np.max(np.abs(np.nan_to_num(sampler.pressure_field))) * 0.8
    plt.clim(-lims, lims)
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, bbox_inches='tight')
        plt.close()


def analyse_semicircle_sampled_data(sampler_angles, sampler_radius,p,distance_between_points):
    """
    Performs analysis on the data calculated by a semicircle sampler

    .. Note:

        This procedure assumes a lot about correctness of the underlying data! Use as per example only.

    Example:

    .. code-block: python

        # create a sampling line, semi-circle around the array, to sample the field there
        tau=2*np.pi
        sampler_radius=focal_radius
        sampler_angles=np.linspace(-tau/4,tau/4,num=2048,endpoint=True)
        ys=np.sin(sampler_angles)*sampler_radius
        zs=np.cos(sampler_angles)*sampler_radius
        xs=np.zeros(ys.shape)

        semicircle_sampler=handybeam_world.add_sampler(handybeam.samplers.clist_sampler.ClistSampler())
        semicircle_sampler.add_sampling_points(xs,ys,zs)

        semicircle_sampler.propagate()
        strathclyde.analise_semicircle_sampled_data(sampler_angles,sampler_radius,semicircle_sampler.pressure_field)

        strathclyde.print_analysis(stats)

    :param np.array() sampler_angles: the x-axis of the data
    :param float sampler_radius: radius of the semicircle sampled from
    :param np.array() p: pressure data
    :return: HandyDict with statistics about the provided data
    """
    # start a dictionary
    stats = HandyDict({'pabs':np.abs(p)})
    # convert p to abs(p)
    stats['pabs'] = np.abs(p)
    # find peak p
    stats['peak_p'] = np.max(stats['pabs'])
    # find location of the peak_p
    stats['peak_location_idx'] = np.argmax(stats['pabs'])
    stats['peak_location_angle'] = sampler_angles[stats['peak_location_idx']]
    # make a db scale vector
    stats['p_db'] = 20.0*np.log10(stats['pabs'])
    stats['p_db'] = stats['p_db']-np.max(stats['p_db'])
    # make a -3dB mask and measure it's width
    stats['db_mask_3db'] = stats['p_db'] > -3.0
    stats['beam_width_idx'] = np.sum(stats['db_mask_3db'])
    stats['first_up'] = np.argwhere(stats['db_mask_3db'])[0][0]
    stats['last_up'] = np.argwhere(stats['db_mask_3db'])[-1][0]
    stats['beam_width_3db_radians'] = sampler_angles[stats['last_up']] - sampler_angles[stats['first_up']]
    stats['beam_width_linear'] = stats['beam_width_3db_radians'] * sampler_radius
    # create a side lobe mask
    last_idx = len(sampler_angles)
    stats['mask_main_lobe_right'] = min(last_idx, stats['last_up']+stats['beam_width_idx'])
    stats['mask_main_lobe_left'] = max(0, stats['first_up']-stats['beam_width_idx'])
    tmp = range(last_idx)
    stats['mask_main_lobe'] = np.bitwise_and(tmp < stats['mask_main_lobe_right'], tmp > stats['mask_main_lobe_left'])
    stats['power_main_lobe'] = np.sum(stats['pabs']**2 * stats['db_mask_3db'])*distance_between_points # note: integration here, need to multiply by dx
    stats['power_side_lobes'] = np.sum(stats['pabs']**2 * ~stats['mask_main_lobe'])* distance_between_points
    stats['peak_sidelobe'] = np.max(stats['pabs']*~stats['mask_main_lobe'])
    stats['contrast_mts_ratio'] = 20*np.log10(stats['power_main_lobe']/stats['power_side_lobes'])
    return stats


def print_analysis(stats):
    print('Main lobe:')
    print(' >> peak amplitude value: {}'.format(stats['peak_p']))
    print(' >> peak location : {:0.3f} radians = {:0.2f} degrees'.format(
        stats['peak_location_angle'],
        stats['peak_location_angle']*180/np.pi))
    print(' >> angular width (-3dB): {:0.3f} radians = {:0.2f} degrees'.format(
        stats['beam_width_3db_radians'],
        stats['beam_width_3db_radians']*180/np.pi))
    print(' >> linear width (-3dB): {:0.3f}mm '.format(stats['beam_width_linear']*1e3))
    print('integrated main lobe power: {}'.format(stats['power_main_lobe']))
    print('Side lobes:')
    print(' >> peak side lobe value :{}'.format(stats['peak_sidelobe']))
    print(' >> integrated side lobe power: {}'.format(stats['power_side_lobes']))
    print('Contrast metric:')
    print(' >> integrated main lobe to side lobe ratio: {:0.2f} dB'.format(stats['contrast_mts_ratio']))
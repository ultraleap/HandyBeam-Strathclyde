"""

module :mod:`strathclyde` -- complementary material for the RCNDE Ultrasonic Transduction Course

Contains the class :class:`LinearArray`.


"""
from os import linesep
import numpy as np
import matplotlib.pyplot as plt

tau = 2 * np.pi

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
                 focal_point=np.array((0.0e-3, 0.0e-3, 200e-3))
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
        """
        self.radiation_frequency = radiation_frequency
        self.sound_velocity = sound_velocity
        self.passive_aperture = passive_aperture
        self.sampling_density = sampling_density
        self.element_pitch = element_pitch
        self.element_count = element_count
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
    def passive_aperture_focus_power_estimage(self):
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
        tofs=np.full((self.element_count,1), np.NaN)  # initialize the output with NaNs
        for idx in range(self.element_count):
            tofs[idx,0] = self.time_of_flight(self.focal_point, locations[idx, :])

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

        :return:
        """
        return np.ones(shape=(self.element_count, 1))

    def create_point_cloud_for_array_element(self,
                                             passive_aperture=np.NaN,
                                             active_aperture=np.NaN,
                                             element_position=np.array((0.0, 0.0, 0.0)),
                                             tx_amplitude=1.0,
                                             tx_phase=0.0):
        """ Creates a cloud of :code:`xyznnnddddap____` points that simulate a larger aperture of a single phased array element.

        for the description of what is `xyznnnddddap____`, see the `tx_array_descriptor_a` documentation of `HandyBeam`

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
             point_ones_list,  # directivity_amplitude_poly2_c0
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
        """

        .. ToDo::

            ! ToDo: check if the direction of phases is correct -- could need to be negative here

        :return: point cloud, numpy.array.shape == (:,16), the count of points generated depends on :code:`self.dx_simulation`


        """
        element_locations = self.element_centre_locations
        focal_laws = self.focal_laws_delays
        # convert time-domain focal law to frequency-domain focal law at the frequency of interest
        # this means, wrap
        # ! ToDo: check if the direction of phases is correct -- could need to be negative here
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
            >>   Active aperture near field transition: 119.4mm

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
        txt = txt + linesep + ">>   Active aperture near field transition: {:0.1f}mm".format(self.active_aperture_near_field*1e3)
        txt = txt + linesep + ">>   Passive aperture near field transition: {:0.1f}mm".format(self.passive_aperture_near_field * 1e3)
        txt = txt + linesep + ">>   Active aperture near field transition: {:0.1f}mm".format(self.active_aperture_near_field*1e3)
        txt = txt + linesep + "  "
        txt = txt + linesep + "> Focal point calculated properties:"
        txt = txt + linesep + ">>   focal distance: {:0.1f}mm".format(self.focal_distance*1e3)
        txt = txt + linesep + ">>   active aperture -6dB focal spot size: {:0.1f}mm".format(self.active_aperture_focus_power_estimate * 1e3)
        txt = txt + linesep + ">>   passive aperture -6dB natural focus spot size: {:0.1f}mm".format(self.passive_aperture_focus_power_estimage * 1e3)
        return txt

    def __repr__(self):
        """links to self.__str__()"""
        return self.__str__()

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
        plt.ylabel('time of flight from probe'+linesep+'to focal point[$\mu$s]')
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
        plt.ylabel("time of flight from probe"+linesep+"to focal point[$\mu$s]")
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



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
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

        .. Todo::

            At this time, all the gains are set to 1.0 -- that is, no apodisation. Write the code to make the apodisation.

        """
        delays = self.focal_laws_delays
        gains = self.focal_laws_gains
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

        .. ToDo::

            Currently a stock array is given to the handybeam.world.World(); modify so that the sampled-array is given with the appropriate focal law and apodisation applied
            use create_point_cloud_for_array_element() and generate_focal_law_phases() and generate_apodisaion_coefficients()

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


def visualize_2d_amplitude_xz(sampler,figsize=(4, 3), dpi=150, filename=None):
    """this visualizes using sampler.coordinates -- that is, coordinates are absolute to the world"""
    plt.figure(figsize=figsize, dpi=dpi)
    limits = get_rectilinear_sampler_coordinates_maxmin(sampler.coordinates)
    plt.imshow(np.abs(sampler.pressure_field), cmap='hot', extent=(limits[2], limits[3], limits[5], limits[4]))
    plt.xlabel('y-coordinate[m]')
    plt.ylabel('z-coordinate[m]')
    lims = np.max(np.abs(np.nan_to_num(sampler.pressure_field)))
    plt.clim(0, lims * 0.8)
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, bbox_inches='tight')
        plt.close()


def visualize_2d_real(sampler, figsize=(4, 3), dpi=150, filename=None):
    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(np.real(sampler.pressure_field), cmap='bwr', extent=sampler.extent)
    plt.xlabel('x-extent of the grid[m]')
    plt.ylabel('y-extent of the grid[m]')
    lims = np.max(np.abs(np.nan_to_num(sampler.pressure_field))) * 0.8
    plt.clim(-lims, lims)
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, bbox_inches='tight')
        plt.close()



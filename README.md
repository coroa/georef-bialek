

# Georeferencing of Bialek model

Janusz Bialek, Qiong Zhou and Neil Hutcheon released a grid model of
the continental European transmission system, see

<http://www.powerworld.com/knowledge-base/updated-and-validated-power-flow-model-of-the-main-continental-european-transmission-network>

and

<http://wiki.openmod-initiative.org/wiki/Transmission_network_datasets#Bialek_European_Model>

for a description of the model and comparison to other models.

The model is "made available for public use" but has no official
licence. It is reproduced here in ./original-data.

Although the model contains impedances and the number of circuits, it
does not have geo-coordinates for the nodes. There are some
coordinates in the PowerWorld model, but they are not longitude and
latitude.

This project is an attempt to convert the PowerWorld model coordinates
into proper geo-coordinates.

Qiong Zhou described the coordinate transformation from ArcGIS to
PowerWorld in pages 136-140 of her PhD thesis

<http://etheses.dur.ac.uk/1263/1/1263.pdf>

but this information has not been sufficient to reconstruct the
geo-coordinates.


The code here makes a few informed guesses, but the transformation are
still not correct (see ./final-data/bialek-transformed.pdf).

The code assumes that the PowerWorld coordinates are linearly related
to the projection used in the ENTSO-E map of the transmission system,
which uses projection [EPSG 3034](http://prj2epsg.org/epsg/3034),
which is a Lambert Conformal Conical projection.

Please help if you have any better ideas!

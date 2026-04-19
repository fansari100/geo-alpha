package io.geoalpha.service.domain;

import jakarta.validation.constraints.DecimalMax;
import jakarta.validation.constraints.DecimalMin;
import jakarta.validation.constraints.NotBlank;

/** WGS-84 axis-aligned bounding box plus a human-readable label. */
public record AreaOfInterest(
        @NotBlank String name,
        @DecimalMin("-90") @DecimalMax("90") double minLatDeg,
        @DecimalMin("-90") @DecimalMax("90") double maxLatDeg,
        @DecimalMin("-180") @DecimalMax("180") double minLonDeg,
        @DecimalMin("-180") @DecimalMax("180") double maxLonDeg
) {
    public double approxAreaKm2() {
        final double avgLat = Math.toRadians((minLatDeg + maxLatDeg) / 2.0);
        return Math.abs(maxLatDeg - minLatDeg) * 111.0
             * Math.abs(maxLonDeg - minLonDeg) * 111.0 * Math.cos(avgLat);
    }
}

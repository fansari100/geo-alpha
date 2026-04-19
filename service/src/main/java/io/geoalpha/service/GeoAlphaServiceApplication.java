package io.geoalpha.service;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableAsync;
import org.springframework.scheduling.annotation.EnableScheduling;

/**
 * Entry point for the geo-alpha mission orchestration service.
 *
 * <p>The orchestrator owns the lifecycle of a "mission" - an analyst
 * request to image a region of interest, run an analytic stack on
 * the result, and ship the structured intelligence product back to
 * the operator console.  The actual analytics live in the Python
 * gateway; this service is the control plane that schedules them
 * and aggregates the responses.
 */
@SpringBootApplication
@EnableScheduling
@EnableAsync
public class GeoAlphaServiceApplication {

    public static void main(String[] args) {
        SpringApplication.run(GeoAlphaServiceApplication.class, args);
    }
}

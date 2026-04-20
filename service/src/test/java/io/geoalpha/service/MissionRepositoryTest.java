package io.geoalpha.service;

import io.geoalpha.service.domain.AreaOfInterest;
import io.geoalpha.service.domain.Mission;
import io.geoalpha.service.domain.MissionRepository;
import org.junit.jupiter.api.Test;

import java.time.Instant;
import java.util.List;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.*;

class MissionRepositoryTest {

    private Mission sample(Mission.Priority p) {
        return new Mission(
                UUID.randomUUID(), "Test", new AreaOfInterest("Site", 33, 33.5, -117.5, -117),
                p, List.of(Mission.Analytic.REGIME_DETECT),
                Instant.now(), null, null, Mission.State.SCHEDULED, "test"
        );
    }

    @Test
    void priorityOrderingHonoured() {
        var repo = new MissionRepository();
        repo.save(sample(Mission.Priority.ROUTINE));
        repo.save(sample(Mission.Priority.FLASH));
        var pending = repo.findByState(Mission.State.SCHEDULED);
        assertEquals(2, pending.size());
        assertEquals(Mission.Priority.FLASH, pending.get(0).priority());
    }

    @Test
    void cancelTransitions() {
        var repo = new MissionRepository();
        var m = sample(Mission.Priority.PRIORITY);
        repo.save(m);
        repo.save(m.withState(Mission.State.CANCELLED));
        assertEquals(Mission.State.CANCELLED, repo.findById(m.id()).orElseThrow().state());
    }
}

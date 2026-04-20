package io.geoalpha.service.controller;

import io.geoalpha.service.domain.IntelProduct;
import io.geoalpha.service.domain.Mission;
import io.geoalpha.service.domain.MissionRepository;

import jakarta.validation.Valid;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.UUID;

@RestController
@RequestMapping("/api/v1/missions")
@CrossOrigin
public class MissionController {

    private final MissionRepository repo;

    public MissionController(MissionRepository repo) {
        this.repo = repo;
    }

    @PostMapping
    @ResponseStatus(HttpStatus.CREATED)
    public Mission create(@Valid @RequestBody Mission incoming) {
        UUID id = incoming.id() != null ? incoming.id() : UUID.randomUUID();
        Mission stored = new Mission(
                id,
                incoming.name(),
                incoming.aoi(),
                incoming.priority(),
                incoming.analytics(),
                Instant.now(),
                null,
                null,
                Mission.State.SCHEDULED,
                incoming.operator() == null ? "system" : incoming.operator()
        );
        return repo.save(stored);
    }

    @GetMapping
    public List<Mission> list(@RequestParam(required = false) Mission.State state) {
        return state == null ? repo.findAll() : repo.findByState(state);
    }

    @GetMapping("/{id}")
    public ResponseEntity<Mission> get(@PathVariable UUID id) {
        return repo.findById(id).map(ResponseEntity::ok).orElse(ResponseEntity.notFound().build());
    }

    @GetMapping("/{id}/product")
    public ResponseEntity<IntelProduct> product(@PathVariable UUID id) {
        return repo.productOf(id).map(ResponseEntity::ok).orElse(ResponseEntity.notFound().build());
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<Void> cancel(@PathVariable UUID id) {
        return repo.findById(id).map(m -> {
            repo.save(m.withState(Mission.State.CANCELLED).completedNow());
            return ResponseEntity.noContent().<Void>build();
        }).orElse(ResponseEntity.notFound().build());
    }

    @GetMapping("/stats")
    public Map<String, Object> stats() {
        return Map.of(
                "total", repo.total(),
                "scheduled", repo.findByState(Mission.State.SCHEDULED).size(),
                "running", repo.findByState(Mission.State.RUNNING).size(),
                "completed", repo.findByState(Mission.State.COMPLETED).size(),
                "failed", repo.findByState(Mission.State.FAILED).size()
        );
    }
}

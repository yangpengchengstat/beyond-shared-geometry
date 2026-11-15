"""Fixed, separately authored context frames; never a random language sample."""
from __future__ import annotations
import hashlib
import json

# Each line is a distinct scenario frame, followed by its paired rendering.
# No sampled names/codes manufacture apparent independent contexts.
FRAME_TEXT = '''A conservator copied one detail from a damaged exhibition card: {fact} The torn edge contains no additional instructions.|The damaged exhibition card supplies a single usable detail. The conservator transcribed it as follows: {fact}
A ferry clerk entered a detail into the crossing log. {fact} The passenger count is recorded elsewhere.|The crossing log separates the passenger count from the detail entered by the ferry clerk: {fact}
The rehearsal coordinator read a note aloud before the musicians arrived. {fact} The note concerns the current arrangement.|Before the musicians arrived, the current arrangement was described in the coordinator's rehearsal note: {fact}
Inside a sealed envelope, the librarian found a catalogue correction. {fact} The correction supersedes the pencilled draft.|The pencilled catalogue draft was superseded by a correction in a sealed envelope. The librarian read: {fact}
A gardener attached a factual label to the irrigation plan. {fact} Water quantities appear on a separate sheet.|The irrigation plan has a factual label distinct from its water-quantity sheet. The gardener's label says: {fact}
The theatre technician checked the approved cue card. {fact} Nothing on the reverse changes that entry.|On the approved cue card checked by the theatre technician, the unchanged entry is: {fact}
The registrar dictated a field from a museum loan form. {fact} The insurance estimate was not part of the dictation.|The museum loan form supplied one dictated field, excluding the insurance estimate. The registrar stated: {fact}
A survey assistant read the annotation beside a boundary sketch. {fact} The scale bar belongs to the sketch itself.|Beside the boundary sketch was an annotation read by the survey assistant: {fact} Its content is separate from the scale bar.
The kitchen manager recorded a detail on the service board. {fact} The ingredient list remains in the pantry.|On the service board, separate from the pantry ingredient list, the kitchen manager recorded: {fact}
A translator preserved a factual entry in the translated notice. {fact} The surrounding sentences explain how to file it.|The translated notice preserves the entry {fact} Its translator treated the surrounding filing instructions separately.
The volunteer responsible for the trail register copied an entry. {fact} The weather description is omitted here.|Excluding the weather description, the trail-register volunteer copied this entry: {fact}
A stage designer placed a concise note next to the scale model. {fact} The note describes the approved version.|The approved version of the scale model is described by the stage designer's adjacent note: {fact}
The boat builder checked a detail in the inspection record. {fact} The material measurements are irrelevant to this entry.|In the inspection record, the boat builder located the entry {fact} It is separate from the material measurements.
A postal assistant read the entry beneath the sorting diagram. {fact} The diagram's title does not modify it.|Beneath the sorting diagram is the entry read by the postal assistant: {fact} It is unchanged by the title.
The astronomy club secretary entered a fact into the observation register. {fact} Equipment costs were listed separately.|The observation register distinguishes equipment costs from the secretary's factual entry: {fact}
A bookbinder saved the instruction card accompanying a restoration job. {fact} This is the card's factual field.|On the saved instruction card for the restoration job, the bookbinder identified the factual field: {fact}
The choir administrator copied a detail from the approved programme. {fact} The composer biography was on another page.|The approved programme provided the choir administrator with this detail: {fact} The biography appears separately.
A custodian transcribed the label on a storage diagram. {fact} The room inventory was not included.|The label on the storage diagram, transcribed without the room inventory, states: {fact}
The ceramics instructor recorded a fact on the workshop card. {fact} The firing temperature is given in another field.|The workshop card has a factual field separate from the firing temperature. The ceramics instructor entered: {fact}
A railway archivist copied a detail from a historical service sheet. {fact} The locomotive description is unrelated.|From the historical service sheet, the railway archivist copied {fact} The locomotive description was excluded.
The projectionist read the approved setup note. {fact} The lens inventory appears in a different document.|The approved setup note, rather than the lens inventory, tells the projectionist: {fact}
A community organiser entered a fact in the hall-use register. {fact} Attendance figures were left blank.|The hall-use register leaves attendance blank but contains the organiser's factual entry: {fact}
The textile curator transcribed a field from the handling card. {fact} Fibre composition was reported separately.|On the handling card, separate from fibre composition, the textile curator transcribed: {fact}
A rowing instructor checked a note in the training folder. {fact} The crew list does not change this note.|The training folder contains the note {fact} The rowing instructor confirmed that it is independent of the crew list.
The bakery supervisor recorded a factual detail on the delivery sheet. {fact} Product quantities were tabulated elsewhere.|On the delivery sheet, the bakery supervisor wrote {fact} Product quantities occupy a separate table.
A printmaker read the factual field on a proof label. {fact} Ink measurements are absent from this excerpt.|The proof label's factual field, read by the printmaker, is {fact} This excerpt excludes ink measurements.
The dental receptionist copied a field from an administrative card. {fact} No clinical information is included.|The administrative card provides the receptionist with the field {fact} It contains no clinical information here.
A puppeteer saved the factual note attached to the scenery plan. {fact} The dialogue belongs to a separate script.|Attached to the scenery plan is the puppeteer's factual note: {fact} The dialogue is recorded separately.
The sports club treasurer read a detail in the facility booking record. {fact} Membership fees are outside this record.|The facility booking record gives the treasurer this detail: {fact} It does not concern membership fees.
A clock repairer entered a fact on a work-order card. {fact} Gear measurements were placed on the reverse.|The work-order card separates reverse-side gear measurements from this entry by the repairer: {fact}
The aquarium educator copied a fact onto the demonstration card. {fact} Species descriptions appear in the guidebook.|On the demonstration card, apart from the guidebook's species descriptions, the educator copied: {fact}
A school caretaker read the entry in a room-preparation note. {fact} The furniture count is not reproduced.|The room-preparation note contains the entry {fact} The caretaker read it without reproducing the furniture count.
The costume assistant wrote a factual annotation on the fitting chart. {fact} Fabric dimensions were stored elsewhere.|The fitting chart's factual annotation reads {fact} The costume assistant kept fabric dimensions separate.
A piano tuner checked the factual field in a service folder. {fact} Acoustic measurements are not shown.|In the service folder, the piano tuner found {fact} The acoustic measurements are omitted from this excerpt.
The cycling event steward copied a fact from the control sheet. {fact} Rider numbers appear in a separate column.|The control sheet gives the cycling steward the fact {fact} Rider numbers are recorded separately.
A film archivist read a detail on the screening instruction card. {fact} The film synopsis is irrelevant here.|The screening instruction card, apart from the film synopsis, supplies the archivist with this detail: {fact}
The pottery cooperative secretary recorded a fact in a shared register. {fact} Sales totals are maintained elsewhere.|In the shared register, the cooperative secretary entered {fact} It is separate from the sales totals.
A bridge inspector copied a factual field from the visit sheet. {fact} Structural measurements were excluded.|The visit sheet contains the field {fact} The bridge inspector copied it without structural measurements.
The dance teacher placed a note beside the rehearsal diagram. {fact} The musical notation is on a separate page.|Beside the rehearsal diagram, the dance teacher's note says {fact} Musical notation appears separately.
A maritime guide transcribed a field from the tour record. {fact} Historical commentary was not copied.|The tour record supplies the maritime guide with the field {fact} The copied material excludes historical commentary.
The instrument maker read the factual part of a workshop notice. {fact} Assembly instructions were in an attachment.|The workshop notice's factual part is {fact} The instrument maker distinguished it from attached assembly instructions.
A seed-bank assistant copied a fact from an accession card. {fact} Germination statistics are reported elsewhere.|On the accession card, apart from germination statistics, the seed-bank assistant copied: {fact}
The chamber ensemble manager entered a fact in a venue folder. {fact} Performer biographies occupy another folder.|The venue folder contains the manager's entry {fact} Performer biographies are filed separately.
A jewellery conservator checked the factual annotation on a handling sheet. {fact} Material analysis is not part of it.|The handling sheet has the annotation {fact} The conservator treats the material analysis separately.
The planetarium presenter read a factual field on a demonstration sheet. {fact} The narration appears in another document.|The demonstration sheet provides the presenter with {fact} Its factual field is distinct from the narration.
A woodworking tutor copied a detail from a studio notice. {fact} Tool specifications are omitted here.|The studio notice's detail is {fact} The woodworking tutor copied it without the tool specifications.
The community radio producer checked an entry in the transmission folder. {fact} Interview transcripts were filed elsewhere.|The transmission folder contains the producer's entry {fact} Interview transcripts are separate.
A landscape architect recorded a fact on a site-visit card. {fact} Plant quantities are not included in this field.|The site-visit card gives the landscape architect this factual field: {fact} Plant quantities appear elsewhere.
The badminton club secretary read a detail from a hall instruction sheet. {fact} Player rankings are unrelated.|The hall instruction sheet supplies the secretary with {fact} This detail does not concern player rankings.
A map restorer transcribed an annotation from a condition report. {fact} The colour analysis was left out.|The condition report includes the annotation {fact} The map restorer transcribed it without colour analysis.
The opera rehearsal assistant copied a factual field from the approved sheet. {fact} Vocal markings remain in the score.|The approved sheet gives the rehearsal assistant the field {fact} Vocal markings are recorded in the score instead.
A waterworks archivist read an entry on an inspection index card. {fact} Flow measurements are not reproduced.|The inspection index card contains the entry {fact} The archivist omitted the flow measurements.
The chess club organiser entered a fact on the room allocation form. {fact} Match results were stored separately.|On the room allocation form, separate from match results, the chess organiser entered: {fact}
A botanical illustrator checked the factual label beside a drawing. {fact} The drawing's dimensions are elsewhere.|Beside the drawing is the label {fact} The botanical illustrator considered its dimensions separately.
The canoe centre administrator copied a field from a logistics note. {fact} Safety instructions remain in the handbook.|The logistics note supplies the administrator with {fact} Safety instructions are documented separately.
A glass artist recorded a factual detail on the display card. {fact} Chemical composition is not part of that detail.|The display card contains the glass artist's detail {fact} Chemical composition is excluded.
The village archivist read the factual entry in a register margin. {fact} The rest of the margin is blank.|In the register margin, the village archivist found {fact} No other entry appears in that margin.
A concert hall attendant copied a fact from a preparation checklist. {fact} The equipment count was on a separate checklist.|The preparation checklist provides the attendant with {fact} Equipment counts are kept separately.
The photography tutor checked an entry in a darkroom notice. {fact} Exposure settings are outside this entry.|The darkroom notice contains the entry {fact} The photography tutor kept exposure settings separate.
A geological collection assistant transcribed a fact from a loan slip. {fact} Mineral descriptions are not reproduced.|On the loan slip, the collection assistant transcribed {fact} Mineral descriptions were excluded.
The fencing instructor read a factual field in a facility record. {fact} Technique descriptions belong to the lesson plan.|The facility record gives the fencing instructor the field {fact} Technique descriptions appear elsewhere.
A toy museum guide copied a detail from a demonstration label. {fact} The toy's history is on another label.|The demonstration label supplies the guide with {fact} The historical account is separate.
The sculpture technician checked a fact in the installation folder. {fact} Weight measurements were recorded elsewhere.|The installation folder contains the technician's factual entry {fact} Weight measurements are separate.
A public library coordinator transcribed a field from a display planning sheet. {fact} Book summaries were not copied.|The display planning sheet supplies the coordinator with {fact} The transcription excludes book summaries.'''

STATES = {'C7': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
          'D4': ['NW', 'NE', 'SE', 'SW']}


def build_bank():
    frames = [line.split('|') for line in FRAME_TEXT.strip().splitlines()]
    if len(frames) != 64 or any(len(pair) != 2 for pair in frames):
        raise ValueError('Expected 64 explicitly authored paired frames')
    rows = []
    for domain, states in STATES.items():
        for i, pair in enumerate(frames):
            family = f'new_context_{i:03d}'
            for para, frame in enumerate(pair):
                if frame.count('{fact}') != 1:
                    raise ValueError('Each frame must contain one fact')
                for state, token in enumerate(states):
                    fact = f'The scheduled weekday is {token}.' if domain == 'C7' else f'The marked square corner is {token}.'
                    text = frame.format(fact=fact)
                    if text.count(token) != 1:
                        raise ValueError('Ambiguous target span')
                    start = text.index(token)
                    rows.append({'row_id': f'{domain}/{family}/p{para}/lexical_{domain}/s{state}',
                                 'family_id': family, 'paraphrase_id': para, 'state_id': state,
                                 'orbit_id': f'lexical_{domain}', 'split': 'test', 'context_split': 'test',
                                 'domain': domain, 'text': text, 'target': token,
                                 'source_span': [start, start + len(token)], 'task': 'structural',
                                 'sampling_scope': 'fixed_assistant_authored_frames_human_review_required',
                                 'authoring_provenance': 'separately_written_for_context_generalization_v1'})
    validate_bank(rows)
    return rows


def validate_bank(rows):
    if len(rows) != 1408:
        raise ValueError('Bank must have 1408 rows')
    ids, texts, blocks = set(), set(), {}
    for row in rows:
        domain = row['domain']; states = STATES[domain]
        if row['row_id'] in ids or row['text'] in texts:
            raise ValueError('Duplicate row or exact prompt')
        ids.add(row['row_id']); texts.add(row['text'])
        a, b = row['source_span']
        if row['text'][a:b] != states[row['state_id']] or row['target'] != states[row['state_id']]:
            raise ValueError('Incorrect target span')
        if row['split'] != 'test' or row['paraphrase_id'] not in (0, 1):
            raise ValueError('Bank must remain test-only and paired')
        blocks.setdefault((domain,row['family_id'],row['paraphrase_id']), []).append(row['state_id'])
    expected = {(d,f'new_context_{i:03d}',p) for d in STATES for i in range(64) for p in (0,1)}
    if set(blocks) != expected or any(sorted(v) != list(range(len(STATES[k[0]]))) for k,v in blocks.items()):
        raise ValueError('Incomplete or repeated state block')
    return {'rows': len(rows), 'families': 64, 'blocks_per_domain':128,
            'human_independence_claim':False, 'population_sampling_claim':False}

#!/usr/bin/env bash
# Pull every SpheroSeg microtubule project flagged `verified` from production (cvat2): the
# annotations of all frames, the ONE static segmentation-source frame per project, and -- for
# projects seeded by the model that is deployed right now -- the proposal that model emits.
#
# Read-only on production: SELECTs, file reads, and one /segment request per project (the same
# request the backend sends when a user clicks Segment; it writes nothing and goes through the
# service's own inference lock). Projects seeded by a model that is no longer deployed need
# scripts/spheroseg_reinfer.py with that model's package and weights instead.
#
# Idempotent: existing proposal files are kept, so a re-run costs no inference.
#
#   scripts/spheroseg_verified_fetch.sh [data/real/spheroseg_verified]
#   PROPOSALS=0 scripts/spheroseg_verified_fetch.sh    # dump + frames only
#
# then:  python -m mt_bench.spheroseg_verified --root data/real/spheroseg_verified
set -euo pipefail

ROOT="${1:-data/real/spheroseg_verified}"
HOST="${SPHEROSEG_HOST:-cvat2}"
UPLOADS=/home/cvat/cell-segmentation-hub/backend/uploads/blue
ML=http://127.0.0.1:4008/api/v1
DEPLOYED_THR="${DEPLOYED_THR:-0.98}"   # prob_thr of the deployed package (params_sparse35.json)
PROPOSALS="${PROPOSALS:-1}"

psql_at() {  # one SQL statement on stdin -> unaligned tuples on stdout
  ssh -o BatchMode=yes "$HOST" 'docker exec -i spheroseg-postgres psql -U spheroseg -d spheroseg -At -F "|" -v ON_ERROR_STOP=1 -f -'
}

mkdir -p "$ROOT/raw/images" "$ROOT/proposals"

# The channel the model was shown: the one flagged as segmentation source, else the hub's own
# fallback, the channel behind the frame row's file (one project reached the hub with its IRM
# channel named TIRF_488_AD and nothing flagged).
SRC_CH=$(cat <<'EOS'
coalesce((select c->>'name' from jsonb_array_elements(i.channels) c
          where (c->>'isSegmentationSource')::boolean limit 1),
         (select regexp_replace(f."originalPath", '^.*/([^/]+)[.]png$', '\1')
          from images f where f."parentVideoId" = i.id and f."frameIndex" = 0))
EOS
)

echo "annotations ..."
psql_at > "$ROOT/raw/projects.json" <<SQL
select json_agg(row_to_json(t)) from (
  select p.id as project_id, p.title, p.description, p."createdAt" as project_created,
         p."verifiedAt" as verified_at, md5(p."userId") as owner_md5, i.id as video_id,
         i.name as container_name, i."originalPath" as container_path, i."fileSize" as container_bytes,
         i.width, i.height, i."frameCount" as frame_count, i."pixelSizeUm" as pixel_size_um,
         i."frameIntervalMs" as frame_interval_ms, i.channels, $SRC_CH as source_channel
  from projects p join images i on i."projectId" = p.id and i."isVideoContainer"
  where p.type = 'microtubules' and p.verified order by p."verifiedAt") t
SQL
ssh -o BatchMode=yes "$HOST" 'docker exec -i spheroseg-postgres psql -U spheroseg -d spheroseg -At -v ON_ERROR_STOP=1 -f - | gzip -6' \
  > "$ROOT/raw/segmentations.jsonl.gz" <<'SQL'
select row_to_json(t) from (
  select p.id as project_id, p.title, p."createdAt" as project_created, p."verifiedAt" as verified_at,
         md5(p."verifiedBy") as verified_by_md5, md5(p."userId") as owner_md5,
         (p."verifiedBy" = p."userId") as verified_by_owner, i.id as image_id,
         i."parentVideoId" as video_id, i."frameIndex" as frame_index, i.name as image_name,
         i."originalPath" as original_path, i.width, i.height, s.id as seg_id, s.model, s.threshold,
         s."createdAt" as seg_created, s."updatedAt" as seg_updated, s."imageWidth" as seg_w,
         s."imageHeight" as seg_h, md5(s.polygons) as polygons_md5, s.polygons::json as polygons
  from projects p join images i on i."projectId" = p.id join segmentations s on s."imageId" = i.id
  where p.type = 'microtubules' and p.verified order by p."verifiedAt", i."frameIndex") t
SQL
gzip -t "$ROOT/raw/segmentations.jsonl.gz"

# project | video dir (relative to the uploads root) | source channel | seed threshold
LIST=$(psql_at <<SQL
select p.id, replace(i."originalPath", '/original.nd2', ''), $SRC_CH,
       (select min(s.threshold) from segmentations s join images f on f.id = s."imageId" where f."projectId" = p.id)
from projects p join images i on i."projectId" = p.id and i."isVideoContainer"
where p.type = 'microtubules' and p.verified order by p."verifiedAt"
SQL
)
echo "$(printf '%s\n' "$LIST" | wc -l | tr -d ' ') verified microtubule projects"

while IFS='|' read -r pid dir ch thr; do
  [ -n "$pid" ] || continue
  rel="$dir/frames/0000/$ch.png"
  # The benchmark unit is one static frame per project. If the source channel ever varies over
  # the movie, that assumption is gone and the project must be looked at before it is used.
  distinct=$(ssh -n -o BatchMode=yes "$HOST" "md5sum $UPLOADS/$dir/frames/*/$ch.png | cut -d' ' -f1 | sort -u | wc -l")
  [ "$distinct" = "1" ] || echo "WARNING ${pid:0:8}: source channel $ch has $distinct distinct frames, not 1"
  ssh -n -o BatchMode=yes "$HOST" "tar cf - -C $UPLOADS/projects '${rel#projects/}'" | tar xf - -C "$ROOT/raw/images"
  local_png="$ROOT/raw/images/${rel#projects/}"
  remote_md5=$(ssh -n -o BatchMode=yes "$HOST" "md5sum '$UPLOADS/$rel' | cut -d' ' -f1")
  local_md5=$( (md5sum "$local_png" 2>/dev/null || md5 -r "$local_png") | cut -d' ' -f1)
  [ "$remote_md5" = "$local_md5" ] || { echo "CHECKSUM MISMATCH $local_png"; exit 1; }
  out="$ROOT/proposals/${pid:0:8}__prod.json"
  if [ "$PROPOSALS" != "1" ] || [ -s "$out" ]; then
    echo "${pid:0:8} $ch ok (thr $thr)"
  elif [ "$thr" = "$DEPLOYED_THR" ]; then
    ssh -n -o BatchMode=yes "$HOST" "curl -sf --max-time 120 -F 'file=@$UPLOADS/$rel' -F model=microtubule $ML/segment" > "$out"
    echo "${pid:0:8} $ch ok, proposal from the deployed model"
  else
    echo "${pid:0:8} $ch ok -- seeded at thr $thr by a model no longer deployed: run scripts/spheroseg_reinfer.py"
  fi
done <<< "$LIST"

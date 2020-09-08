// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: ortools/constraint_solver/demon_profiler.proto

#include "ortools/constraint_solver/demon_profiler.pb.h"

#include <algorithm>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/wire_format_lite.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
extern PROTOBUF_INTERNAL_EXPORT_ortools_2fconstraint_5fsolver_2fdemon_5fprofiler_2eproto ::PROTOBUF_NAMESPACE_ID::internal::SCCInfo<0> scc_info_DemonRuns_ortools_2fconstraint_5fsolver_2fdemon_5fprofiler_2eproto;
namespace operations_research {
class DemonRunsDefaultTypeInternal {
 public:
  ::PROTOBUF_NAMESPACE_ID::internal::ExplicitlyConstructed<DemonRuns> _instance;
} _DemonRuns_default_instance_;
class ConstraintRunsDefaultTypeInternal {
 public:
  ::PROTOBUF_NAMESPACE_ID::internal::ExplicitlyConstructed<ConstraintRuns> _instance;
} _ConstraintRuns_default_instance_;
}  // namespace operations_research
static void InitDefaultsscc_info_ConstraintRuns_ortools_2fconstraint_5fsolver_2fdemon_5fprofiler_2eproto() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::operations_research::_ConstraintRuns_default_instance_;
    new (ptr) ::operations_research::ConstraintRuns();
    ::PROTOBUF_NAMESPACE_ID::internal::OnShutdownDestroyMessage(ptr);
  }
  ::operations_research::ConstraintRuns::InitAsDefaultInstance();
}

::PROTOBUF_NAMESPACE_ID::internal::SCCInfo<1> scc_info_ConstraintRuns_ortools_2fconstraint_5fsolver_2fdemon_5fprofiler_2eproto =
    {{ATOMIC_VAR_INIT(::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase::kUninitialized), 1, 0, InitDefaultsscc_info_ConstraintRuns_ortools_2fconstraint_5fsolver_2fdemon_5fprofiler_2eproto}, {
      &scc_info_DemonRuns_ortools_2fconstraint_5fsolver_2fdemon_5fprofiler_2eproto.base,}};

static void InitDefaultsscc_info_DemonRuns_ortools_2fconstraint_5fsolver_2fdemon_5fprofiler_2eproto() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::operations_research::_DemonRuns_default_instance_;
    new (ptr) ::operations_research::DemonRuns();
    ::PROTOBUF_NAMESPACE_ID::internal::OnShutdownDestroyMessage(ptr);
  }
  ::operations_research::DemonRuns::InitAsDefaultInstance();
}

::PROTOBUF_NAMESPACE_ID::internal::SCCInfo<0> scc_info_DemonRuns_ortools_2fconstraint_5fsolver_2fdemon_5fprofiler_2eproto =
    {{ATOMIC_VAR_INIT(::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase::kUninitialized), 0, 0, InitDefaultsscc_info_DemonRuns_ortools_2fconstraint_5fsolver_2fdemon_5fprofiler_2eproto}, {}};

static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_ortools_2fconstraint_5fsolver_2fdemon_5fprofiler_2eproto[2];
static constexpr ::PROTOBUF_NAMESPACE_ID::EnumDescriptor const** file_level_enum_descriptors_ortools_2fconstraint_5fsolver_2fdemon_5fprofiler_2eproto = nullptr;
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_ortools_2fconstraint_5fsolver_2fdemon_5fprofiler_2eproto = nullptr;

const ::PROTOBUF_NAMESPACE_ID::uint32 TableStruct_ortools_2fconstraint_5fsolver_2fdemon_5fprofiler_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::operations_research::DemonRuns, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::operations_research::DemonRuns, demon_id_),
  PROTOBUF_FIELD_OFFSET(::operations_research::DemonRuns, start_time_),
  PROTOBUF_FIELD_OFFSET(::operations_research::DemonRuns, end_time_),
  PROTOBUF_FIELD_OFFSET(::operations_research::DemonRuns, failures_),
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::operations_research::ConstraintRuns, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::operations_research::ConstraintRuns, constraint_id_),
  PROTOBUF_FIELD_OFFSET(::operations_research::ConstraintRuns, initial_propagation_start_time_),
  PROTOBUF_FIELD_OFFSET(::operations_research::ConstraintRuns, initial_propagation_end_time_),
  PROTOBUF_FIELD_OFFSET(::operations_research::ConstraintRuns, failures_),
  PROTOBUF_FIELD_OFFSET(::operations_research::ConstraintRuns, demons_),
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, sizeof(::operations_research::DemonRuns)},
  { 9, -1, sizeof(::operations_research::ConstraintRuns)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::operations_research::_DemonRuns_default_instance_),
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::operations_research::_ConstraintRuns_default_instance_),
};

const char descriptor_table_protodef_ortools_2fconstraint_5fsolver_2fdemon_5fprofiler_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n.ortools/constraint_solver/demon_profil"
  "er.proto\022\023operations_research\"U\n\tDemonRu"
  "ns\022\020\n\010demon_id\030\001 \001(\t\022\022\n\nstart_time\030\002 \003(\003"
  "\022\020\n\010end_time\030\003 \003(\003\022\020\n\010failures\030\004 \001(\003\"\267\001\n"
  "\016ConstraintRuns\022\025\n\rconstraint_id\030\001 \001(\t\022&"
  "\n\036initial_propagation_start_time\030\002 \003(\003\022$"
  "\n\034initial_propagation_end_time\030\003 \003(\003\022\020\n\010"
  "failures\030\004 \001(\003\022.\n\006demons\030\005 \003(\0132\036.operati"
  "ons_research.DemonRunsb\006proto3"
  ;
static const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable*const descriptor_table_ortools_2fconstraint_5fsolver_2fdemon_5fprofiler_2eproto_deps[1] = {
};
static ::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase*const descriptor_table_ortools_2fconstraint_5fsolver_2fdemon_5fprofiler_2eproto_sccs[2] = {
  &scc_info_ConstraintRuns_ortools_2fconstraint_5fsolver_2fdemon_5fprofiler_2eproto.base,
  &scc_info_DemonRuns_ortools_2fconstraint_5fsolver_2fdemon_5fprofiler_2eproto.base,
};
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_ortools_2fconstraint_5fsolver_2fdemon_5fprofiler_2eproto_once;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_ortools_2fconstraint_5fsolver_2fdemon_5fprofiler_2eproto = {
  false, false, descriptor_table_protodef_ortools_2fconstraint_5fsolver_2fdemon_5fprofiler_2eproto, "ortools/constraint_solver/demon_profiler.proto", 350,
  &descriptor_table_ortools_2fconstraint_5fsolver_2fdemon_5fprofiler_2eproto_once, descriptor_table_ortools_2fconstraint_5fsolver_2fdemon_5fprofiler_2eproto_sccs, descriptor_table_ortools_2fconstraint_5fsolver_2fdemon_5fprofiler_2eproto_deps, 2, 0,
  schemas, file_default_instances, TableStruct_ortools_2fconstraint_5fsolver_2fdemon_5fprofiler_2eproto::offsets,
  file_level_metadata_ortools_2fconstraint_5fsolver_2fdemon_5fprofiler_2eproto, 2, file_level_enum_descriptors_ortools_2fconstraint_5fsolver_2fdemon_5fprofiler_2eproto, file_level_service_descriptors_ortools_2fconstraint_5fsolver_2fdemon_5fprofiler_2eproto,
};

// Force running AddDescriptors() at dynamic initialization time.
static bool dynamic_init_dummy_ortools_2fconstraint_5fsolver_2fdemon_5fprofiler_2eproto = (static_cast<void>(::PROTOBUF_NAMESPACE_ID::internal::AddDescriptors(&descriptor_table_ortools_2fconstraint_5fsolver_2fdemon_5fprofiler_2eproto)), true);
namespace operations_research {

// ===================================================================

void DemonRuns::InitAsDefaultInstance() {
}
class DemonRuns::_Internal {
 public:
};

DemonRuns::DemonRuns(::PROTOBUF_NAMESPACE_ID::Arena* arena)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena),
  start_time_(arena),
  end_time_(arena) {
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:operations_research.DemonRuns)
}
DemonRuns::DemonRuns(const DemonRuns& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      start_time_(from.start_time_),
      end_time_(from.end_time_) {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  demon_id_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  if (!from._internal_demon_id().empty()) {
    demon_id_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), from._internal_demon_id(),
      GetArena());
  }
  failures_ = from.failures_;
  // @@protoc_insertion_point(copy_constructor:operations_research.DemonRuns)
}

void DemonRuns::SharedCtor() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&scc_info_DemonRuns_ortools_2fconstraint_5fsolver_2fdemon_5fprofiler_2eproto.base);
  demon_id_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  failures_ = PROTOBUF_LONGLONG(0);
}

DemonRuns::~DemonRuns() {
  // @@protoc_insertion_point(destructor:operations_research.DemonRuns)
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

void DemonRuns::SharedDtor() {
  GOOGLE_DCHECK(GetArena() == nullptr);
  demon_id_.DestroyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
}

void DemonRuns::ArenaDtor(void* object) {
  DemonRuns* _this = reinterpret_cast< DemonRuns* >(object);
  (void)_this;
}
void DemonRuns::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void DemonRuns::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}
const DemonRuns& DemonRuns::default_instance() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&::scc_info_DemonRuns_ortools_2fconstraint_5fsolver_2fdemon_5fprofiler_2eproto.base);
  return *internal_default_instance();
}


void DemonRuns::Clear() {
// @@protoc_insertion_point(message_clear_start:operations_research.DemonRuns)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  start_time_.Clear();
  end_time_.Clear();
  demon_id_.ClearToEmpty(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), GetArena());
  failures_ = PROTOBUF_LONGLONG(0);
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* DemonRuns::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  ::PROTOBUF_NAMESPACE_ID::Arena* arena = GetArena(); (void)arena;
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    CHK_(ptr);
    switch (tag >> 3) {
      // string demon_id = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 10)) {
          auto str = _internal_mutable_demon_id();
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParser(str, ptr, ctx);
          CHK_(::PROTOBUF_NAMESPACE_ID::internal::VerifyUTF8(str, "operations_research.DemonRuns.demon_id"));
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // repeated int64 start_time = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 18)) {
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::PackedInt64Parser(_internal_mutable_start_time(), ptr, ctx);
          CHK_(ptr);
        } else if (static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 16) {
          _internal_add_start_time(::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr));
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // repeated int64 end_time = 3;
      case 3:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 26)) {
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::PackedInt64Parser(_internal_mutable_end_time(), ptr, ctx);
          CHK_(ptr);
        } else if (static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 24) {
          _internal_add_end_time(::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr));
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // int64 failures = 4;
      case 4:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 32)) {
          failures_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      default: {
      handle_unusual:
        if ((tag & 7) == 4 || tag == 0) {
          ctx->SetLastTag(tag);
          goto success;
        }
        ptr = UnknownFieldParse(tag,
            _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(),
            ptr, ctx);
        CHK_(ptr != nullptr);
        continue;
      }
    }  // switch
  }  // while
success:
  return ptr;
failure:
  ptr = nullptr;
  goto success;
#undef CHK_
}

::PROTOBUF_NAMESPACE_ID::uint8* DemonRuns::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:operations_research.DemonRuns)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // string demon_id = 1;
  if (this->demon_id().size() > 0) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->_internal_demon_id().data(), static_cast<int>(this->_internal_demon_id().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "operations_research.DemonRuns.demon_id");
    target = stream->WriteStringMaybeAliased(
        1, this->_internal_demon_id(), target);
  }

  // repeated int64 start_time = 2;
  {
    int byte_size = _start_time_cached_byte_size_.load(std::memory_order_relaxed);
    if (byte_size > 0) {
      target = stream->WriteInt64Packed(
          2, _internal_start_time(), byte_size, target);
    }
  }

  // repeated int64 end_time = 3;
  {
    int byte_size = _end_time_cached_byte_size_.load(std::memory_order_relaxed);
    if (byte_size > 0) {
      target = stream->WriteInt64Packed(
          3, _internal_end_time(), byte_size, target);
    }
  }

  // int64 failures = 4;
  if (this->failures() != 0) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt64ToArray(4, this->_internal_failures(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:operations_research.DemonRuns)
  return target;
}

size_t DemonRuns::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:operations_research.DemonRuns)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // repeated int64 start_time = 2;
  {
    size_t data_size = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      Int64Size(this->start_time_);
    if (data_size > 0) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
            static_cast<::PROTOBUF_NAMESPACE_ID::int32>(data_size));
    }
    int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(data_size);
    _start_time_cached_byte_size_.store(cached_size,
                                    std::memory_order_relaxed);
    total_size += data_size;
  }

  // repeated int64 end_time = 3;
  {
    size_t data_size = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      Int64Size(this->end_time_);
    if (data_size > 0) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
            static_cast<::PROTOBUF_NAMESPACE_ID::int32>(data_size));
    }
    int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(data_size);
    _end_time_cached_byte_size_.store(cached_size,
                                    std::memory_order_relaxed);
    total_size += data_size;
  }

  // string demon_id = 1;
  if (this->demon_id().size() > 0) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
        this->_internal_demon_id());
  }

  // int64 failures = 4;
  if (this->failures() != 0) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int64Size(
        this->_internal_failures());
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    return ::PROTOBUF_NAMESPACE_ID::internal::ComputeUnknownFieldsSize(
        _internal_metadata_, total_size, &_cached_size_);
  }
  int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void DemonRuns::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:operations_research.DemonRuns)
  GOOGLE_DCHECK_NE(&from, this);
  const DemonRuns* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<DemonRuns>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:operations_research.DemonRuns)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:operations_research.DemonRuns)
    MergeFrom(*source);
  }
}

void DemonRuns::MergeFrom(const DemonRuns& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:operations_research.DemonRuns)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  start_time_.MergeFrom(from.start_time_);
  end_time_.MergeFrom(from.end_time_);
  if (from.demon_id().size() > 0) {
    _internal_set_demon_id(from._internal_demon_id());
  }
  if (from.failures() != 0) {
    _internal_set_failures(from._internal_failures());
  }
}

void DemonRuns::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:operations_research.DemonRuns)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void DemonRuns::CopyFrom(const DemonRuns& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:operations_research.DemonRuns)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool DemonRuns::IsInitialized() const {
  return true;
}

void DemonRuns::InternalSwap(DemonRuns* other) {
  using std::swap;
  _internal_metadata_.Swap<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(&other->_internal_metadata_);
  start_time_.InternalSwap(&other->start_time_);
  end_time_.InternalSwap(&other->end_time_);
  demon_id_.Swap(&other->demon_id_, &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), GetArena());
  swap(failures_, other->failures_);
}

::PROTOBUF_NAMESPACE_ID::Metadata DemonRuns::GetMetadata() const {
  return GetMetadataStatic();
}


// ===================================================================

void ConstraintRuns::InitAsDefaultInstance() {
}
class ConstraintRuns::_Internal {
 public:
};

ConstraintRuns::ConstraintRuns(::PROTOBUF_NAMESPACE_ID::Arena* arena)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena),
  initial_propagation_start_time_(arena),
  initial_propagation_end_time_(arena),
  demons_(arena) {
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:operations_research.ConstraintRuns)
}
ConstraintRuns::ConstraintRuns(const ConstraintRuns& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      initial_propagation_start_time_(from.initial_propagation_start_time_),
      initial_propagation_end_time_(from.initial_propagation_end_time_),
      demons_(from.demons_) {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  constraint_id_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  if (!from._internal_constraint_id().empty()) {
    constraint_id_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), from._internal_constraint_id(),
      GetArena());
  }
  failures_ = from.failures_;
  // @@protoc_insertion_point(copy_constructor:operations_research.ConstraintRuns)
}

void ConstraintRuns::SharedCtor() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&scc_info_ConstraintRuns_ortools_2fconstraint_5fsolver_2fdemon_5fprofiler_2eproto.base);
  constraint_id_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  failures_ = PROTOBUF_LONGLONG(0);
}

ConstraintRuns::~ConstraintRuns() {
  // @@protoc_insertion_point(destructor:operations_research.ConstraintRuns)
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

void ConstraintRuns::SharedDtor() {
  GOOGLE_DCHECK(GetArena() == nullptr);
  constraint_id_.DestroyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
}

void ConstraintRuns::ArenaDtor(void* object) {
  ConstraintRuns* _this = reinterpret_cast< ConstraintRuns* >(object);
  (void)_this;
}
void ConstraintRuns::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void ConstraintRuns::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}
const ConstraintRuns& ConstraintRuns::default_instance() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&::scc_info_ConstraintRuns_ortools_2fconstraint_5fsolver_2fdemon_5fprofiler_2eproto.base);
  return *internal_default_instance();
}


void ConstraintRuns::Clear() {
// @@protoc_insertion_point(message_clear_start:operations_research.ConstraintRuns)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  initial_propagation_start_time_.Clear();
  initial_propagation_end_time_.Clear();
  demons_.Clear();
  constraint_id_.ClearToEmpty(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), GetArena());
  failures_ = PROTOBUF_LONGLONG(0);
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* ConstraintRuns::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  ::PROTOBUF_NAMESPACE_ID::Arena* arena = GetArena(); (void)arena;
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    CHK_(ptr);
    switch (tag >> 3) {
      // string constraint_id = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 10)) {
          auto str = _internal_mutable_constraint_id();
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParser(str, ptr, ctx);
          CHK_(::PROTOBUF_NAMESPACE_ID::internal::VerifyUTF8(str, "operations_research.ConstraintRuns.constraint_id"));
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // repeated int64 initial_propagation_start_time = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 18)) {
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::PackedInt64Parser(_internal_mutable_initial_propagation_start_time(), ptr, ctx);
          CHK_(ptr);
        } else if (static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 16) {
          _internal_add_initial_propagation_start_time(::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr));
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // repeated int64 initial_propagation_end_time = 3;
      case 3:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 26)) {
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::PackedInt64Parser(_internal_mutable_initial_propagation_end_time(), ptr, ctx);
          CHK_(ptr);
        } else if (static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 24) {
          _internal_add_initial_propagation_end_time(::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr));
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // int64 failures = 4;
      case 4:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 32)) {
          failures_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // repeated .operations_research.DemonRuns demons = 5;
      case 5:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 42)) {
          ptr -= 1;
          do {
            ptr += 1;
            ptr = ctx->ParseMessage(_internal_add_demons(), ptr);
            CHK_(ptr);
            if (!ctx->DataAvailable(ptr)) break;
          } while (::PROTOBUF_NAMESPACE_ID::internal::ExpectTag<42>(ptr));
        } else goto handle_unusual;
        continue;
      default: {
      handle_unusual:
        if ((tag & 7) == 4 || tag == 0) {
          ctx->SetLastTag(tag);
          goto success;
        }
        ptr = UnknownFieldParse(tag,
            _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(),
            ptr, ctx);
        CHK_(ptr != nullptr);
        continue;
      }
    }  // switch
  }  // while
success:
  return ptr;
failure:
  ptr = nullptr;
  goto success;
#undef CHK_
}

::PROTOBUF_NAMESPACE_ID::uint8* ConstraintRuns::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:operations_research.ConstraintRuns)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // string constraint_id = 1;
  if (this->constraint_id().size() > 0) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->_internal_constraint_id().data(), static_cast<int>(this->_internal_constraint_id().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "operations_research.ConstraintRuns.constraint_id");
    target = stream->WriteStringMaybeAliased(
        1, this->_internal_constraint_id(), target);
  }

  // repeated int64 initial_propagation_start_time = 2;
  {
    int byte_size = _initial_propagation_start_time_cached_byte_size_.load(std::memory_order_relaxed);
    if (byte_size > 0) {
      target = stream->WriteInt64Packed(
          2, _internal_initial_propagation_start_time(), byte_size, target);
    }
  }

  // repeated int64 initial_propagation_end_time = 3;
  {
    int byte_size = _initial_propagation_end_time_cached_byte_size_.load(std::memory_order_relaxed);
    if (byte_size > 0) {
      target = stream->WriteInt64Packed(
          3, _internal_initial_propagation_end_time(), byte_size, target);
    }
  }

  // int64 failures = 4;
  if (this->failures() != 0) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt64ToArray(4, this->_internal_failures(), target);
  }

  // repeated .operations_research.DemonRuns demons = 5;
  for (unsigned int i = 0,
      n = static_cast<unsigned int>(this->_internal_demons_size()); i < n; i++) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessage(5, this->_internal_demons(i), target, stream);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:operations_research.ConstraintRuns)
  return target;
}

size_t ConstraintRuns::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:operations_research.ConstraintRuns)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // repeated int64 initial_propagation_start_time = 2;
  {
    size_t data_size = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      Int64Size(this->initial_propagation_start_time_);
    if (data_size > 0) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
            static_cast<::PROTOBUF_NAMESPACE_ID::int32>(data_size));
    }
    int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(data_size);
    _initial_propagation_start_time_cached_byte_size_.store(cached_size,
                                    std::memory_order_relaxed);
    total_size += data_size;
  }

  // repeated int64 initial_propagation_end_time = 3;
  {
    size_t data_size = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      Int64Size(this->initial_propagation_end_time_);
    if (data_size > 0) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
            static_cast<::PROTOBUF_NAMESPACE_ID::int32>(data_size));
    }
    int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(data_size);
    _initial_propagation_end_time_cached_byte_size_.store(cached_size,
                                    std::memory_order_relaxed);
    total_size += data_size;
  }

  // repeated .operations_research.DemonRuns demons = 5;
  total_size += 1UL * this->_internal_demons_size();
  for (const auto& msg : this->demons_) {
    total_size +=
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(msg);
  }

  // string constraint_id = 1;
  if (this->constraint_id().size() > 0) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
        this->_internal_constraint_id());
  }

  // int64 failures = 4;
  if (this->failures() != 0) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int64Size(
        this->_internal_failures());
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    return ::PROTOBUF_NAMESPACE_ID::internal::ComputeUnknownFieldsSize(
        _internal_metadata_, total_size, &_cached_size_);
  }
  int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void ConstraintRuns::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:operations_research.ConstraintRuns)
  GOOGLE_DCHECK_NE(&from, this);
  const ConstraintRuns* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<ConstraintRuns>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:operations_research.ConstraintRuns)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:operations_research.ConstraintRuns)
    MergeFrom(*source);
  }
}

void ConstraintRuns::MergeFrom(const ConstraintRuns& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:operations_research.ConstraintRuns)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  initial_propagation_start_time_.MergeFrom(from.initial_propagation_start_time_);
  initial_propagation_end_time_.MergeFrom(from.initial_propagation_end_time_);
  demons_.MergeFrom(from.demons_);
  if (from.constraint_id().size() > 0) {
    _internal_set_constraint_id(from._internal_constraint_id());
  }
  if (from.failures() != 0) {
    _internal_set_failures(from._internal_failures());
  }
}

void ConstraintRuns::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:operations_research.ConstraintRuns)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void ConstraintRuns::CopyFrom(const ConstraintRuns& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:operations_research.ConstraintRuns)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool ConstraintRuns::IsInitialized() const {
  return true;
}

void ConstraintRuns::InternalSwap(ConstraintRuns* other) {
  using std::swap;
  _internal_metadata_.Swap<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(&other->_internal_metadata_);
  initial_propagation_start_time_.InternalSwap(&other->initial_propagation_start_time_);
  initial_propagation_end_time_.InternalSwap(&other->initial_propagation_end_time_);
  demons_.InternalSwap(&other->demons_);
  constraint_id_.Swap(&other->constraint_id_, &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), GetArena());
  swap(failures_, other->failures_);
}

::PROTOBUF_NAMESPACE_ID::Metadata ConstraintRuns::GetMetadata() const {
  return GetMetadataStatic();
}


// @@protoc_insertion_point(namespace_scope)
}  // namespace operations_research
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::operations_research::DemonRuns* Arena::CreateMaybeMessage< ::operations_research::DemonRuns >(Arena* arena) {
  return Arena::CreateMessageInternal< ::operations_research::DemonRuns >(arena);
}
template<> PROTOBUF_NOINLINE ::operations_research::ConstraintRuns* Arena::CreateMaybeMessage< ::operations_research::ConstraintRuns >(Arena* arena) {
  return Arena::CreateMessageInternal< ::operations_research::ConstraintRuns >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>

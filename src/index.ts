import {
  AudioPlayerStatus,
  VoiceConnectionStatus,
  entersState,
  joinVoiceChannel,
} from "@discordjs/voice";
import {
  ChatInputCommandInteraction,
  Client,
  Events,
  GatewayIntentBits,
  GuildMember,
  MessageFlags,
  PermissionFlagsBits,
  REST,
  Routes,
  SlashCommandBuilder,
} from "discord.js";
import { config } from "./config.js";
import { CoreEventClient } from "./core-events.js";
import { splitDiscordMessage } from "./discord-messages.js";
import {
  getCoreHealth,
  getTtsVoices,
  grantWebSearchAccess,
  listWebSearchAccess,
  resetConversation,
  revokeWebSearchAccess,
  setTtsEffect,
  setTtsVoice,
  syncUserDirectory,
} from "./core-client.js";
import { logError, logInfo } from "./logger.js";
import { VoiceSession } from "./voice-session.js";

const MAX_IMAGE_BYTES = 10 * 1024 * 1024;
const SUPPORTED_IMAGE_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);

const client = new Client({
  intents: [GatewayIntentBits.Guilds, GatewayIntentBits.GuildVoiceStates],
});
const sessions = new Map<string, VoiceSession>();
const coreEvents = new CoreEventClient((event) => {
  const session = sessions.get(event.guildId);
  if (!session || session.channelId !== event.channelId) return;
  if (event.type === "chat_message") {
    void sendVoiceChatMessage(event.channelId, event.content ?? "").catch((error) => {
      logError("voice.chat_message.failed", error, {
        guildId: event.guildId,
        channelId: event.channelId,
        userId: event.userId,
      });
    });
    return;
  }
  session.handleCoreEvent(event);
});

async function sendVoiceChatMessage(channelId: string, content: string): Promise<void> {
  const channel = await client.channels.fetch(channelId);
  if (!channel?.isSendable()) {
    throw new Error(`Discord channel ${channelId} does not support messages`);
  }
  const parts = splitDiscordMessage(content);
  for (const part of parts) {
    await channel.send({
      content: part,
      allowedMentions: { parse: [] },
    });
  }
  logInfo("voice.chat_message.sent", { channelId, parts: parts.length, chars: content.length });
}

async function registerCommands(): Promise<void> {
  const applicationId = client.application?.id ?? client.user?.id;
  if (!applicationId) throw new Error("Discord application ID is unavailable after login");
  const voiceGuildId = config.discordGuildId ?? client.guilds.cache.first()?.id;
  if (!voiceGuildId) throw new Error("Bot is not connected to a Discord guild");
  const tts = await getTtsVoices(voiceGuildId);
  const voiceChoices = tts.voices.slice(0, 25).map((voice) => ({
    name: voice.label,
    value: voice.id,
  }));
  const effectChoices = tts.effects.slice(0, 25).map((effect) => ({
    name: effect.label,
    value: effect.id,
  }));
  const commands = [
    new SlashCommandBuilder().setName("join").setDescription("Подключить ассистента к вашему voice-каналу"),
    new SlashCommandBuilder().setName("leave").setDescription("Отключить ассистента от voice-канала"),
    new SlashCommandBuilder().setName("reset").setDescription("Очистить историю текущего диалога"),
    new SlashCommandBuilder().setName("status").setDescription("Проверить состояние voice-core"),
    new SlashCommandBuilder()
      .setName("image")
      .setDescription("Добавить изображение к следующему голосовому запросу")
      .addAttachmentOption((option) =>
        option.setName("image").setDescription("JPEG, PNG или WebP до 10 МБ").setRequired(true),
      ),
    new SlashCommandBuilder()
      .setName("voice")
      .setDescription("Переключить модель и голос озвучки")
      .addStringOption((option) =>
        option
          .setName("model")
          .setDescription("Модель озвучки")
          .setRequired(true)
          .addChoices(...voiceChoices),
      ),
    new SlashCommandBuilder()
      .setName("effect")
      .setDescription("Переключить обработку голоса")
      .addStringOption((option) =>
        option
          .setName("profile")
          .setDescription("Профиль обработки")
          .setRequired(true)
          .addChoices(...effectChoices),
      ),
    new SlashCommandBuilder()
      .setName("web-access")
      .setDescription("Управлять доступом к поиску в сети")
      .setDefaultMemberPermissions(PermissionFlagsBits.Administrator)
      .addSubcommand((subcommand) =>
        subcommand
          .setName("add")
          .setDescription("Разрешить пользователю поиск в сети")
          .addUserOption((option) =>
            option.setName("user").setDescription("Пользователь").setRequired(true),
          ),
      )
      .addSubcommand((subcommand) =>
        subcommand
          .setName("remove")
          .setDescription("Убрать пользователя из списка доступа")
          .addUserOption((option) =>
            option.setName("user").setDescription("Пользователь").setRequired(true),
          ),
      )
      .addSubcommand((subcommand) =>
        subcommand.setName("list").setDescription("Показать список доступа"),
      ),
  ].map((command) => command.toJSON());
  const rest = new REST().setToken(config.discordToken);
  if (config.discordGuildId) {
    if (!client.guilds.cache.has(config.discordGuildId)) {
      const availableGuilds = [...client.guilds.cache.values()]
        .map((guild) => `${guild.name} (${guild.id})`)
        .join(", ");
      throw new Error(
        `Bot cannot access DISCORD_GUILD_ID=${config.discordGuildId}. ` +
          `Invite it to that server or use one of: ${availableGuilds || "no accessible servers"}`,
      );
    }
    await rest.put(Routes.applicationGuildCommands(applicationId, config.discordGuildId), {
      body: commands,
    });
    return;
  }
  await rest.put(Routes.applicationCommands(applicationId), { body: commands });
}

async function handleJoin(interaction: ChatInputCommandInteraction): Promise<void> {
  const member = interaction.member as GuildMember | null;
  const voiceChannel = member?.voice.channel;
  if (!interaction.guild || !voiceChannel) {
    await interaction.reply({
      content: "Сначала войдите в voice-канал.",
      flags: MessageFlags.Ephemeral,
    });
    return;
  }

  await interaction.deferReply({ flags: MessageFlags.Ephemeral });
  const guildId = interaction.guild.id;
  logInfo("voice.join.requested", {
    guildId,
    channelId: voiceChannel.id,
    channelName: voiceChannel.name,
    userId: interaction.user.id,
  });
  sessions.get(guildId)?.close("replaced_by_join");
  const connection = joinVoiceChannel({
    channelId: voiceChannel.id,
    guildId,
    adapterCreator: interaction.guild.voiceAdapterCreator,
    selfDeaf: false,
    selfMute: true,
  });
  connection.on("stateChange", (oldState, newState) => {
    logInfo("voice.connection.state", {
      guildId,
      channelId: voiceChannel.id,
      previousState: oldState.status,
      state: newState.status,
    });
    if (
      newState.status === VoiceConnectionStatus.Disconnected ||
      newState.status === VoiceConnectionStatus.Destroyed
    ) {
      logInfo("voice.disconnected", {
        guildId,
        channelId: voiceChannel.id,
        state: newState.status,
      });
      if (newState.status === VoiceConnectionStatus.Destroyed) {
        sessions.delete(guildId);
      }
    }
  });

  try {
    await entersState(connection, VoiceConnectionStatus.Ready, 20_000);
    const session = new VoiceSession(client, connection, voiceChannel.id);
    sessions.set(guildId, session);
    void syncUserDirectory(
      guildId,
      [...voiceChannel.members.values()]
        .filter((voiceMember) => voiceMember.id !== client.user?.id)
        .map((voiceMember) => ({
          userId: voiceMember.id,
          displayName: voiceMember.displayName,
        })),
    ).catch((error) => {
      logError("voice.user_directory.sync_failed", error, {
        guildId,
        channelId: voiceChannel.id,
      });
    });
    logInfo("voice.connected", {
      guildId,
      channelId: voiceChannel.id,
      channelName: voiceChannel.name,
    });
    await interaction.editReply("Подключился. Можно говорить.");
  } catch (error) {
    connection.destroy();
    logError("voice.connection.failed", error, {
      guildId,
      channelId: voiceChannel.id,
    });
    await interaction.editReply("Не удалось подключиться к voice-каналу.");
  }
}

async function handleImage(interaction: ChatInputCommandInteraction): Promise<void> {
  const guildId = interaction.guildId;
  const member = interaction.member as GuildMember | null;
  const session = guildId ? sessions.get(guildId) : undefined;
  if (!guildId || !session || member?.voice.channelId !== session.channelId) {
    await interaction.reply({
      content: "Сначала подключите бота через /join и войдите в тот же voice-канал.",
      flags: MessageFlags.Ephemeral,
    });
    return;
  }

  const attachment = interaction.options.getAttachment("image", true);
  const contentType = attachment.contentType?.toLowerCase() ?? "";
  if (!SUPPORTED_IMAGE_TYPES.has(contentType)) {
    await interaction.reply({
      content: "Поддерживаются только JPEG, PNG и WebP.",
      flags: MessageFlags.Ephemeral,
    });
    return;
  }
  if (attachment.size > MAX_IMAGE_BYTES) {
    await interaction.reply({
      content: "Изображение должно быть не больше 10 МБ.",
      flags: MessageFlags.Ephemeral,
    });
    return;
  }

  await interaction.deferReply({ flags: MessageFlags.Ephemeral });
  const response = await fetch(attachment.url, { signal: AbortSignal.timeout(15_000) });
  if (!response.ok) throw new Error(`Discord attachment download returned ${response.status}`);
  const data = Buffer.from(await response.arrayBuffer());
  if (data.length > MAX_IMAGE_BYTES) throw new Error("Discord attachment exceeded 10 MB");

  session.armImagePrompt(interaction.user.id, {
    data,
    contentType,
    name: attachment.name,
  });
  await interaction.editReply("Изображение добавлено. После сигнала надиктуйте запрос.");
}

async function handleWebAccess(interaction: ChatInputCommandInteraction): Promise<void> {
  if (!interaction.guildId || !interaction.guild) {
    throw new Error("This command is only available in a guild");
  }
  if (!interaction.memberPermissions?.has(PermissionFlagsBits.Administrator)) {
    await interaction.reply({
      content: "Команда доступна только администраторам сервера.",
      flags: MessageFlags.Ephemeral,
    });
    return;
  }

  const action = interaction.options.getSubcommand(true);
  await interaction.deferReply({ flags: MessageFlags.Ephemeral });
  if (action === "list") {
    const users = await listWebSearchAccess(interaction.guildId);
    const content = users.length
      ? users.map((user) => `• ${user.displayName ?? user.userId}`).join("\n")
      : "Список пуст. Администраторы имеют доступ автоматически.";
    const [firstPart, ...remainingParts] = splitDiscordMessage(content);
    await interaction.editReply(firstPart);
    for (const part of remainingParts) {
      await interaction.followUp({ content: part, flags: MessageFlags.Ephemeral });
    }
    return;
  }

  const user = interaction.options.getUser("user", true);
  const member = await interaction.guild.members.fetch(user.id).catch(() => null);
  const displayName = member?.displayName ?? user.globalName ?? user.username;
  if (action === "add") {
    await grantWebSearchAccess(interaction.guildId, user.id, displayName);
    await interaction.editReply(`Доступ к поиску выдан: ${displayName}.`);
    return;
  }
  await revokeWebSearchAccess(interaction.guildId, user.id);
  await interaction.editReply(`Доступ к поиску отозван: ${displayName}.`);
}

async function handleInteraction(interaction: ChatInputCommandInteraction): Promise<void> {
  const guildId = interaction.guildId;
  switch (interaction.commandName) {
    case "join":
      await handleJoin(interaction);
      break;
    case "leave":
      if (guildId) {
        logInfo("voice.leave.requested", { guildId, userId: interaction.user.id });
        sessions.get(guildId)?.close("leave_command");
      }
      if (guildId) sessions.delete(guildId);
      await interaction.reply({ content: "Отключился.", flags: MessageFlags.Ephemeral });
      break;
    case "reset":
      if (!guildId) throw new Error("This command is only available in a guild");
      await resetConversation(guildId);
      await interaction.reply({
        content: "История диалога очищена.",
        flags: MessageFlags.Ephemeral,
      });
      break;
    case "status": {
      const status = await getCoreHealth();
      const session = guildId ? sessions.get(guildId) : undefined;
      const audio = session?.player.state.status ?? AudioPlayerStatus.Idle;
      await interaction.reply({
        content: `voice-core: ${status}; audio: ${audio}`,
        flags: MessageFlags.Ephemeral,
      });
      break;
    }
    case "image":
      await handleImage(interaction);
      break;
    case "voice": {
      if (!guildId) throw new Error("This command is only available in a guild");
      const voiceId = interaction.options.getString("model", true);
      await interaction.deferReply({ flags: MessageFlags.Ephemeral });
      const voice = await setTtsVoice(guildId, voiceId);
      await interaction.editReply(`Озвучка переключена: ${voice.label}.`);
      break;
    }
    case "effect": {
      if (!guildId) throw new Error("This command is only available in a guild");
      const effectId = interaction.options.getString("profile", true);
      await interaction.deferReply({ flags: MessageFlags.Ephemeral });
      const effect = await setTtsEffect(guildId, effectId);
      await interaction.editReply(`Обработка голоса: ${effect.label}.`);
      break;
    }
    case "web-access":
      await handleWebAccess(interaction);
      break;
  }
}

client.once(Events.ClientReady, async () => {
  logInfo("discord.ready", { bot: client.user?.tag, botId: client.user?.id });
  coreEvents.start();
  try {
    await registerCommands();
    logInfo("discord.commands.registered", { guildId: config.discordGuildId ?? "global" });
  } catch (error) {
    logError("discord.commands.registration_failed", error);
  }
});

client.on("interactionCreate", (interaction) => {
  if (!interaction.isChatInputCommand()) return;
  void handleInteraction(interaction).catch(async (error) => {
    logError("discord.command.failed", error, {
      command: interaction.commandName,
      guildId: interaction.guildId,
      userId: interaction.user.id,
    });
    const message = "Команда завершилась с ошибкой. Проверьте логи бота.";
    try {
      if (interaction.replied || interaction.deferred) await interaction.editReply(message);
      else await interaction.reply({ content: message, flags: MessageFlags.Ephemeral });
    } catch (responseError) {
      logError("discord.command.error_response_failed", responseError, {
        command: interaction.commandName,
        guildId: interaction.guildId,
        userId: interaction.user.id,
      });
    }
  });
});

client.on(Events.VoiceStateUpdate, (_oldState, newState) => {
  const session = sessions.get(newState.guild.id);
  const member = newState.member;
  if (
    !session ||
    newState.channelId !== session.channelId ||
    !member ||
    member.id === client.user?.id
  ) {
    return;
  }
  void syncUserDirectory(newState.guild.id, [
    { userId: member.id, displayName: member.displayName },
  ]).catch((error) => {
    logError("voice.user_directory.sync_failed", error, {
      guildId: newState.guild.id,
      channelId: session.channelId,
      userId: member.id,
    });
  });
});

let shuttingDown = false;

async function shutdown(signal: "SIGINT" | "SIGTERM"): Promise<void> {
  if (shuttingDown) return;
  shuttingDown = true;
  logInfo("discord.shutdown.requested", { signal, voiceSessions: sessions.size });
  for (const session of sessions.values()) session.close("shutdown");
  sessions.clear();
  coreEvents.stop();

  // Keep the Discord gateway alive briefly so voice state updates reach Discord.
  await new Promise((resolve) => setTimeout(resolve, 500));
  client.destroy();
  logInfo("discord.shutdown.completed", { signal });
}

process.once("SIGINT", () => void shutdown("SIGINT"));
process.once("SIGTERM", () => void shutdown("SIGTERM"));

await client.login(config.discordToken);

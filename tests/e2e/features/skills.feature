Feature: Skills
  Verifies: .ai/spec/provider-contract.md (Skill discovery, Tool execution)
  The system discovers skills from the mounted directory, runs their
  scripts, and serves non-skill queries normally when skills are present.

  Scenario: System starts and serves requests with skills mounted
    Given the sandbox service is running with skills
    And a simple non-skill query has been prepared
    When I run the agent with the prepared query and no output schema
    Then the run completes successfully
    And the response has a non-empty summary
    And success is true

  Scenario: System runs a skill and returns structured output
    Given the sandbox service is running with skills
    And the echo-token skill query has been prepared
    When I run the agent with the prepared echo-token query
    Then the batch job completes
    And the skill script wrote a token file to disk
    And the response JSON validates against the output schema
    And the response contains the generated token

  Scenario: Single-file skill script receives its requested argument
    Given the sandbox service is running with skills
    And the single-file skill script task has been prepared
    When I run the agent with the prepared skill-script query
    Then the run completes successfully
    And the response JSON validates against the output schema
    And the response contains the generated skill-script result

  Scenario: Skill command chain shares setup state in one shell
    Given the sandbox service is running with skills
    And the command-chain skill script task has been prepared
    When I run the agent with the prepared skill-script query
    Then the run completes successfully
    And the response JSON validates against the output schema
    And the response contains the generated skill-script result

  Scenario: Non-skill query is unaffected by mounted skills
    Given the sandbox service is running with skills
    And no output schema will be sent
    When I run the agent with the prepared query and no output schema
    Then the run completes successfully
    And the response has a non-empty summary
    And success is true
